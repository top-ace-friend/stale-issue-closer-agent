"""LangGraph graph that uses the GitHub GraphQL API to propose closing stale issues.

With a human-in-the-loop review step using Agent Inbox-compatible interrupts.

Environment variables used:
- GITHUB_TOKEN: Required for GitHub Models and the GitHub GraphQL API.
- TARGET_REPO: Optional. Full repo (e.g. "owner/name").
- Model selection:
    - API_HOST: "github" (default) or "azure".
    - When API_HOST=github: GITHUB_MODEL (default: "gpt-4o").
    - When API_HOST=azure: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_CHAT_DEPLOYMENT, AZURE_OPENAI_VERSION, and either AZURE_OPENAI_API_KEY or Azure AD auth.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import azure.identity
from dotenv import load_dotenv
from jinja2 import BaseLoader, Environment
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langgraph.errors import GraphRecursionError
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.interrupt import (
    ActionRequest,
    HumanInterrupt,
    HumanInterruptConfig,
    HumanResponse,
)
from langgraph.types import interrupt
from pydantic import BaseModel, Field, SecretStr

from agent.github_client import GitHubClient

# Load env vars from a .env file if present
load_dotenv(override=True)

logger = logging.getLogger(__name__)


# Jinja2 environment shared by all templates
_jinja_env = Environment(loader=BaseLoader(), autoescape=False, trim_blocks=True, lstrip_blocks=True)

# Singleton GitHub client (lazy-init) to reuse label cache and HTTP configuration
_GH_CLIENT: GitHubClient | None = None

def get_client() -> GitHubClient:
    global _GH_CLIENT
    if _GH_CLIENT is None:
        _GH_CLIENT = GitHubClient()
    return _GH_CLIENT

# Helper to fetch labels via a shared client instance (no module-level cache needed now)
async def _get_repository_labels() -> dict[str, dict[str, str]]:
    client = get_client()
    try:
        labels = await client.get_repository_labels(TARGET_REPO)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed fetching repository labels for %s: %s", TARGET_REPO, exc)
        return {}
    return {
        (lbl.get("name") or ""): {
            "description": (lbl.get("description") or "").strip(),
            "color": (lbl.get("color") or "").strip(),
        }
        for lbl in labels if lbl.get("name")
    }


# Review description template for human interrupt (required, Jinja2)
REVIEW_TEMPLATE_PATH = Path(__file__).with_name("review_template.md.jinja2")
REVIEW_TEMPLATE = REVIEW_TEMPLATE_PATH.read_text(encoding="utf-8")
REVIEW_TEMPLATE_JINJA = _jinja_env.from_string(REVIEW_TEMPLATE)

# Research phase prompt template (Jinja2)
RESEARCH_PROMPT_TEMPLATE_PATH = Path(__file__).with_name("research_prompt.md.jinja2")
RESEARCH_PROMPT_TEMPLATE = RESEARCH_PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
RESEARCH_PROMPT_TEMPLATE_JINJA = _jinja_env.from_string(RESEARCH_PROMPT_TEMPLATE)

# Propose action system prompt template (Jinja2)
PROPOSE_ACTION_PROMPT_TEMPLATE_PATH = Path(__file__).with_name("propose_action_prompt.md.jinja2")
PROPOSE_ACTION_PROMPT_TEMPLATE = PROPOSE_ACTION_PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
PROPOSE_ACTION_PROMPT_TEMPLATE_JINJA = _jinja_env.from_string(PROPOSE_ACTION_PROMPT_TEMPLATE)

@dataclass
class State:
    """State for single-issue processing."""

    issue: dict[str, Any] | None = None  # Selected issue details (number, title, body, etc.)
    proposal: dict[str, Any] | None = None  # Single proposal
    decision: dict[str, Any] | None = None  # Decision for this issue
    review_note: str | None = None  # Optional note from human review
    research_summary: str | None = None  # Consolidated research notes from first agent


class ProposalModel(BaseModel):
    """Structured proposal produced after research phase."""
    # Identity fields (number, title, url) are intentionally omitted to avoid duplication
    # with State.issue. This model focuses solely on recommended maintenance actions.
    close_issue: bool = Field(description="Whether the issue should be closed now")
    close_issue_rationale: str | None = Field(default=None, description="Specific justification (evidence) for closing or leaving open; cite signals.")
    add_labels: list[str] = Field(
        default_factory=list,
        description="Labels to add (must be existing repository labels; see provided label list with descriptions).",
    )
    add_labels_rationale: str | None = Field(default=None, description="Justification for each label added; cite signals.")
    remove_labels: list[str] = Field(
        default_factory=list,
        description="Labels to remove (must be existing repository labels).",
    )
    remove_labels_rationale: str | None = Field(default=None, description="Justification for each label removed; cite signals or conflicts.")
    assign_issue_to_copilot: bool = Field(description="Whether to assign the issue to the Copilot agent because it appears trivial with an easy / well-bounded fix")
    assign_issue_to_copilot_rationale: str | None = Field(default=None, description="Evidence that the issue is trivial and suitable for automated Copilot agent fix, or why not.")
    post_comment: str | None = Field(
        default=None, description="Optional comment to post (explanatory or request for info)"
    )
    rationale: str = Field(description="Concise reasoning for the chosen actions")


def _build_llm() -> Any:
    """Create the LLM once based on env (GitHub Models or Azure OpenAI)."""
    # Select LLM: GitHub Models (default) or Azure OpenAI
    api_host = os.getenv("API_HOST", "github").lower()
    if api_host == "azure":
        azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
        azure_deployment = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]
        azure_version = os.getenv("AZURE_OPENAI_VERSION", "2024-02-15-preview")

        # If an API key is provided, prefer it
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if api_key:
            return AzureChatOpenAI(
                azure_endpoint=azure_endpoint,
                azure_deployment=azure_deployment,
                api_version=azure_version,
                api_key=SecretStr(api_key),
            )

        # Otherwise, use Azure AD token provider via Azure Developer CLI credential
        token_provider = azure.identity.get_bearer_token_provider(
            azure.identity.AzureDeveloperCliCredential(tenant_id=os.getenv("AZURE_TENANT_ID")),
            "https://cognitiveservices.azure.com/.default",
        )
        return AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=azure_version,
            azure_ad_token_provider=token_provider,
        )
    # Configure the LLM via GitHub Models endpoint
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise RuntimeError(
            "GITHUB_TOKEN is required. Set it to a token that can access GitHub Models and GitHub GraphQL API."
        )
    return ChatOpenAI(
        model=os.getenv("GITHUB_MODEL", "gpt-4o"),
        base_url="https://models.inference.ai.azure.com",
        api_key=SecretStr(token),
    )


@tool("search_issues")
async def tool_search_issues(query: str, config: RunnableConfig) -> str:  # type: ignore[override]
    """Search issues in the target repo, optionally filtering the active issue.

    Context management: If the invoking runnable passes a configurable value
    `active_issue_number` via the `RunnableConfig` (config["configurable"]["active_issue_number"]) then
    that issue will be excluded from the returned list. This avoids self-referential
    results during research.

    Args:
        query: Free-text and/or qualifiers. The repo qualifier is added automatically.
        config: Runtime config (used here only for optional filtering).

    Returns:
        JSON array of issue dicts.
    """
    client = get_client()
    # Over-fetch by 1 so that if the active issue appears we can still return up to 5 others.
    raw_items = await client.search_issues_with_bodies(
        TARGET_REPO, query_text=query, max_results=6, include_comments=True
    )
    active_issue_number = (config or {}).get("configurable", {}).get("active_issue_number")
    filtered = [it for it in raw_items if it.get("number") != active_issue_number]
    items = filtered[:5]
    return json.dumps(items)


@tool("search_code")
async def tool_search_code(query: str) -> str:
    """Search code in the target repo.

    - Do NOT use boolean operators (AND, OR, NOT) in the query. The GitHub code search API does not support them.
    - Use simple keyword or phrase queries only. For example:
        - "form recognizer"
        - "Document Intelligence"
        - "documentanalysis"
        - "DocumentAnalysisClient"
    - If you want to search for multiple keywords, run separate queries for each keyword or phrase.

    Args:
        query: Free-text and/or qualifiers. The repo qualifier is added automatically.

    Returns:
        JSON string containing a list of code hit dicts with repository, path, is_binary, byte_size, and optionally text.
    """
    client = get_client()
    hits = await client.search_codebase(
        TARGET_REPO, query_text=query, max_results=10, include_text=False
    )
    return json.dumps([h.__dict__ for h in hits])

@tool("search_pull_requests")
async def tool_search_pull_requests(query: str) -> str:
    """Search pull requests in the target repo.

    Guidance:
    - The repo qualifier (repo:owner/name) and is:pr are added automatically; don't include them.
    - You may supply additional qualifiers like: `is:open`, `author:login`, `label:bug`, `draft:true`.
    - Use this to find related implementation discussions, recent changes, or precedent decisions.

    Args:
        query: Free-text and/or qualifiers (without the repo qualifier or is:pr).

    Returns:
    JSON list of pull request objects (up to 5). Each contains: id, number, title, url,
    body (description), merged (bool), mergedAt, createdAt, updatedAt.
    """
    client = get_client()
    pulls = await client.search_pull_requests(
    TARGET_REPO, query_text=query, max_results=5
    )
    return json.dumps(pulls)

@tool("fetch_file")
async def tool_fetch_file(filename_or_path: str) -> str:
    """Fetch a file from the repository.

    Args:
        filename_or_path: The full or partial path to the file to fetch.

    Returns:
        JSON string containing the file contents or metadata.
    """
    client = get_client()
    item = await client.fetch_file(TARGET_REPO, filename_or_path, include_text=True)
    return json.dumps(item.__dict__ if item else {})

@tool("get_issue")
async def tool_get_issue(issue_number: int) -> str:
    """Fetch a single issue (and recent comments) by its number.

    Args:
        issue_number: The numeric issue identifier (e.g., 1536).

    Returns:
        JSON string with issue fields (id, number, title, url, state, updatedAt, createdAt, author, labels, body, comments[]).
        Returns an empty JSON object if the issue is not found.
    """
    client = get_client()
    issue = await client.get_issue(TARGET_REPO, issue_number, include_comments=True, comments_limit=50)
    return json.dumps(issue or {})

@tool("get_pull_request")
async def tool_get_pull_request(pr_number: int) -> str:
    """Fetch a single pull request by its number.

    Returns JSON with: number, title, url, description, merged, mergedAt.
    Empty JSON object if not found.
    """
    client = get_client()
    try:
        pr = await client.get_pull_request(TARGET_REPO, pr_number)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": str(exc)})
    return json.dumps(pr or {})

@tool("list_repository_files")
async def tool_list_repository_files(ref: str | None = None) -> str:
    """List all file paths in the target repository at a ref (default branch if omitted).

    Args:
        ref: Optional branch / tag / commit SHA. If not provided, resolves to default branch.

    Returns:
        JSON array of file path strings. May be incomplete for very large repos (tree truncation).
    """
    client = get_client()
    paths = await client.list_repository_files(TARGET_REPO, ref=ref)
    return json.dumps(paths)


async def select_stale_issue_node(state: State, config: RunnableConfig) -> dict[str, Any]:
    """Fetch stale issues and pick one (currently first in list) and its comments.

    Returns partial state with `issue` containing the selected issue, plus embedded comments list.
    """
    client = GitHubClient()
    stale_issues = await client.find_stale_open_issues(TARGET_REPO)
    if not stale_issues:
        raise RuntimeError(f"No stale issues found in {TARGET_REPO}.")
    selected = stale_issues[0]
    comments = await client.get_issue_comments(TARGET_REPO, selected.number)
    issue_dict: dict[str, Any] = {
        "number": selected.number,
        "title": selected.title,
        "url": selected.url,
        "updated_at": selected.updated_at,
        "created_at": selected.created_at,
        "author": selected.author,
        "labels": selected.labels,
        "body": selected.body,
        "comments": comments,
    }
    return {"issue": issue_dict}


async def research_issue_node(state: State, config: RunnableConfig) -> dict[str, Any]:
    """Investigate the selected issue using tools and produce a research summary (NOT the final proposal).

    This node performs exploratory searches/code lookups and synthesizes findings into a plain text
    research_summary that a second node will consume to decide on closure.
    """
    assert state.issue is not None, "Issue must be selected before research."
    issue = state.issue
    model = _build_llm()
    research_tools = [
        tool_search_issues,
        tool_search_code,
        tool_search_pull_requests,
        tool_get_pull_request,
        tool_fetch_file,
        tool_get_issue,
        tool_list_repository_files,
    ]
    agent = create_react_agent(model, research_tools)
    all_comments = issue.get("comments", [])
    prompt = RESEARCH_PROMPT_TEMPLATE_JINJA.render(
        repo=TARGET_REPO,
        issue={
            "number": issue["number"],
            "title": issue["title"],
            "url": issue["url"],
            "updated_at": issue["updated_at"],
            "labels": issue.get("labels") or [],
            "body": issue.get("body") or "",
            "comments": all_comments,
        },
        comments_json=json.dumps(all_comments, ensure_ascii=False),
    )

    # Allow caller or env to override recursion limit (default 20)
    try:
        rec_limit = int(os.getenv("RESEARCH_RECURSION_LIMIT", "20"))
    except ValueError:
        rec_limit = 20
    # Merge provided config with our recursion limit (without mutating original)
    merged_config: RunnableConfig = {
        **(config or {}),
        "configurable": {
            "recursion_limit": rec_limit,
            "active_issue_number": issue["number"],
        },
    }

    try:
        result = await agent.ainvoke({"messages": [("human", prompt)]}, config=merged_config)
    except GraphRecursionError:
        logger.warning("Research agent hit recursion limit (%s). Falling back to minimal summary.", rec_limit)
        fallback = (
            "Research phase exceeded recursion limit before converging. Proceeding with available static context.\n\n"
            "Key Facts:\n"
            f"Issue #{issue['number']}: {issue['title']}\n"
            f"Labels: {', '.join(issue['labels']) if issue.get('labels') else '(none)'}\n"
            f"Last Updated: {issue['updated_at']}\n\n"
            "Signals Against Closing:\n- Agent could not complete research automatically.\n\n"
            "Open Questions:\n- Additional related issues or code references were not gathered due to recursion limit.\n"
        )
        return {"research_summary": fallback}

    messages = result.get("messages") or []
    # Find last AI message without tool calls (research summary)
    for m in reversed(messages):
        if isinstance(m, AIMessage) and not m.tool_calls and isinstance(m.content, str):
            summary = m.content.strip()
            if not summary:
                continue
            return {"research_summary": summary}
    raise RuntimeError("Failed to obtain research summary (no final AI message without tool calls).")


async def propose_action_node(state: State, config: RunnableConfig) -> dict[str, Any]:
    """Take the research summary + issue details and produce a structured proposal.

    Uses a second LLM call with structured output (Option 2 pattern from docs).
    """
    assert state.issue is not None, "Issue must be selected before proposal."
    assert state.research_summary is not None, "Research summary required before proposal."
    issue = state.issue
    research_summary = state.research_summary
    model = _build_llm()
    # Use with_structured_output to force schema
    structured = model.with_structured_output(ProposalModel)
    # Provide only the distilled research to minimize tokens
    # Always fetch the authenticated viewer login via GitHub API (no env var usage)
    gh = get_client()
    maintainer_username = await gh.get_viewer_login()
    label_meta = await _get_repository_labels()
    system_prompt = PROPOSE_ACTION_PROMPT_TEMPLATE_JINJA.render(
        maintainer_username=maintainer_username,
        repo_labels=[{"name": n, "description": meta.get("description", "")} for n, meta in sorted(label_meta.items())],
    )
    user_content = (
        f"Issue number: {issue['number']}\n"
        f"Title: {issue['title']}\n"
        f"URL: {issue['url']}\n"
        f"UpdatedAt: {issue['updated_at']}\n"
        f"Labels: {', '.join(issue['labels']) if issue.get('labels') else '(none)'}\n\n"
        "Research summary:\n" + research_summary
    )
    response: ProposalModel = structured.invoke([
        ("system", system_prompt),
        ("human", user_content),
    ])
    proposal_dict = response.model_dump()
    return {"proposal": proposal_dict}


async def review_issue_node(state: State, config: RunnableConfig) -> dict[str, Any]:
    """Interrupt for the single issue to approve/edit/skip posting a close comment."""
    assert state.proposal is not None, "Proposal must be parsed before review."
    # Identity lives in state.issue; proposed actions live in state.proposal
    assert state.issue is not None, "Issue must be present during review."
    proposal_actions = state.proposal
    issue_identity = state.issue
    number = issue_identity.get("number")
    title = issue_identity.get("title")
    url = issue_identity.get("url")
    close_issue = proposal_actions.get("close_issue")
    add_labels = proposal_actions.get("add_labels") or []
    remove_labels = proposal_actions.get("remove_labels") or []
    post_comment = proposal_actions.get("post_comment")
    rationale = proposal_actions.get("rationale")
    close_issue_rationale = proposal_actions.get("close_issue_rationale")
    add_labels_rationale = proposal_actions.get("add_labels_rationale")
    remove_labels_rationale = proposal_actions.get("remove_labels_rationale")
    assign_issue_to_copilot = proposal_actions.get("assign_issue_to_copilot")
    assign_issue_to_copilot_rationale = proposal_actions.get("assign_issue_to_copilot_rationale")

    actions_for_template = [
        {"name": "close_issue", "value": close_issue, "rationale": close_issue_rationale},
        {"name": "add_labels", "value": add_labels, "rationale": add_labels_rationale},
        {"name": "remove_labels", "value": remove_labels, "rationale": remove_labels_rationale},
        {"name": "assign_issue_to_copilot", "value": assign_issue_to_copilot, "rationale": assign_issue_to_copilot_rationale},
        {"name": "post_comment", "value": bool(post_comment), "rationale": (post_comment[:140] + "…") if post_comment and len(post_comment) > 140 else post_comment},
    ]
    description = REVIEW_TEMPLATE_JINJA.render(
        number=number,
        title=title,
        url=url,
        actions=actions_for_template,
        overall_rationale=rationale or "(none)",
    )

    # Provide only editable action fields (not identity) to reduce accidental edits.
    action_request = ActionRequest(
        action=f"Review maintenance actions: {title}",
        args={
            "close_issue": close_issue,
            "add_labels": add_labels,
            "remove_labels": remove_labels,
            "assign_issue_to_copilot": assign_issue_to_copilot,
            "post_comment": post_comment,
        },
    )
    interrupt_config = HumanInterruptConfig(
        allow_accept=True,
        allow_edit=True,
        allow_respond=True,
        allow_ignore=True,
    )
    request = HumanInterrupt(
        action_request=action_request,
        config=interrupt_config,
        description=description,
    )

    human_response: HumanResponse = interrupt([request])[0]

    # Prepare a decision record for this issue
    decision: dict[str, Any] = {
        "number": number,
        "approved": False,
        "close_issue": bool(close_issue),
        "add_labels": list(add_labels),
        "remove_labels": list(remove_labels),
        "assign_issue_to_copilot": bool(assign_issue_to_copilot),
        "post_comment": post_comment,
        "note": None,
    }

    rtype = human_response.get("type")
    if rtype in ("accept", "edit"):
        ar_raw: Any = human_response.get("args")
        if isinstance(ar_raw, dict):
            inner = ar_raw.get("args")
            ar_args: dict[str, Any] = inner if isinstance(inner, dict) else {}
        else:
            ar_args = {}
        def _coerce_bool(val, default=False):
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                return val.lower() in ("true", "1", "yes")
            return default
        decision.update({
            "approved": True,
            "close_issue": _coerce_bool(ar_args.get("close_issue", close_issue), close_issue),
            "add_labels": ar_args.get("add_labels", add_labels) or [],
            "remove_labels": ar_args.get("remove_labels", remove_labels) or [],
            "assign_issue_to_copilot": _coerce_bool(ar_args.get("assign_issue_to_copilot", assign_issue_to_copilot), assign_issue_to_copilot),
            "post_comment": ar_args.get("post_comment", post_comment),
        })
    elif rtype == "response":
        note = human_response.get("args")
        if isinstance(note, str):
            decision["note"] = note
    elif rtype == "ignore":
        decision["note"] = "ignored"

    return {"decision": decision}


async def finalize_node(state: State, config: RunnableConfig) -> dict[str, Any]:
    """No-op finalize for single-issue mode."""
    return {}


async def apply_decision_node(state: State, config: RunnableConfig) -> dict[str, Any]:
    """Apply approved maintenance actions: labels, comment, close."""
    assert state.proposal is not None and state.decision is not None
    decision = state.decision
    if not decision.get("approved"):
        return {}

    number = int(decision.get("number"))
    close_issue = bool(decision.get("close_issue"))
    add_labels = decision.get("add_labels") or []
    remove_labels = decision.get("remove_labels") or []
    assign_issue_to_copilot = bool(decision.get("assign_issue_to_copilot"))
    post_comment = (decision.get("post_comment") or "").strip() or None

    client = get_client()

    label_meta = await _get_repository_labels()
    allowed = set(label_meta.keys())
    def _normalize_list(vals):
        if not isinstance(vals, list):
            return []
        out: list[str] = []
        for v in vals:
            if not isinstance(v, str):
                continue
            v_clean = v.strip()
            if not v_clean:
                continue
            if v_clean in allowed:
                out.append(v_clean)
            else:
                logger.warning("Ignoring unsupported label '%s' (dynamic allowed list)", v)
        return out
    add_labels = _normalize_list(add_labels)
    remove_labels = _normalize_list(remove_labels)

    # Remove labels first
    for label in remove_labels:
        try:
            await client.remove_label(TARGET_REPO, number, label)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to remove label '%s' on #%s: %s", label, number, exc)

    # Add labels (skip any that were also in remove list to avoid churn)
    for label in add_labels:
        if label in remove_labels:
            continue
        try:
            await client.add_label(TARGET_REPO, number, label)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to add label '%s' to #%s: %s", label, number, exc)

    # 3. Post comment if provided
    if post_comment:
        try:
            await client.post_comment(TARGET_REPO, number, post_comment)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to post comment on #%s: %s", number, exc)

    # 4. Assign to Copilot if requested (before potentially closing to keep it open for automated work)
    if assign_issue_to_copilot and not close_issue:
        try:
            await client.assign_issue_to_copilot(TARGET_REPO, number)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to assign Copilot on #%s: %s", number, exc)

    # 5. Close issue if requested
    if close_issue:
        try:
            await client.close_issue(TARGET_REPO, number)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to close issue #%s: %s", number, exc)

    return {}


TARGET_REPO = os.getenv("TARGET_REPO", "Azure-samples/azure-search-openai-demo")
# Define a new graph
workflow = StateGraph(State)

# Add nodes (single linear flow)
workflow.add_node("select_issue", select_stale_issue_node)
workflow.add_node("research_issue", research_issue_node)
workflow.add_node("propose_action", propose_action_node)
workflow.add_node("review_issue", review_issue_node)
workflow.add_node("apply_decision", apply_decision_node)
workflow.add_node("finalize", finalize_node)

# Linear edges
workflow.add_edge("__start__", "select_issue")
workflow.add_edge("select_issue", "research_issue")
workflow.add_edge("research_issue", "propose_action")
workflow.add_edge("propose_action", "review_issue")
workflow.add_edge("review_issue", "apply_decision")
workflow.add_edge("apply_decision", "finalize")

# Compile the workflow into an executable graph
graph = workflow.compile()
graph.name = "Single Issue Closure"  # Custom name in LangSmith
