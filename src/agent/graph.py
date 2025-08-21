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

import datetime as _dt
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import azure.identity
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.interrupt import (
    ActionRequest,
    HumanInterrupt,
    HumanInterruptConfig,
    HumanResponse,
)
from langgraph.types import interrupt
from pydantic import SecretStr

from agent.github_client import GitHubClient

# Load env vars from a .env file if present
load_dotenv(override=True)

logger = logging.getLogger(__name__)

# Preload the stale prompt template at import time (one-time blocking IO outside async paths)
PROMPT_TEMPLATE_PATH = Path(__file__).with_name("staleprompt.md")
try:
    STALE_PROMPT_TEMPLATE = PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
except Exception as exc:  # noqa: BLE001
    raise RuntimeError(
        f"Failed to read prompt template at {PROMPT_TEMPLATE_PATH}: {exc}"
    ) from exc

@dataclass
class State:
    """State for single-issue processing."""

    proposal: dict[str, Any] | None = None  # Single proposal
    decision: dict[str, Any] | None = None  # Decision for this issue
    review_note: str | None = None  # Optional note from human review


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


@tool("ProposalResponse")
def ProposalResponse(
    number: int,
    title: str,
    url: str,
    can_close: bool,
    rationale: str,
    suggested_comment: str | None = None,
) -> str:
    """Call this tool at the end to emit the final structured proposal.

    Returns the proposal as a JSON string for downstream parsing.
    """
    payload = {
        "proposal": {
            "number": number,
            "title": title,
            "url": url,
            "can_close": can_close,
            "rationale": rationale,
        },
    }
    if suggested_comment is not None:
        payload["proposal"]["suggested_comment"] = suggested_comment
    return json.dumps(payload, ensure_ascii=False)


@tool("search_issues")
async def tool_search_issues(query: str) -> str:
    """Search issues in the target repo.

    Args:
        query: Free-text and/or qualifiers. The repo qualifier is added automatically.

    Returns:
        JSON string containing a list of issue dicts with id, number, title, url, state,
        updatedAt, createdAt, author, labels, body, and optionally comments.
    """
    client = GitHubClient()
    items = await client.search_issues_with_bodies(
        TARGET_REPO, query_text=query, max_results=5, include_comments=True
    )
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
    client = GitHubClient()
    hits = await client.search_codebase(
        TARGET_REPO, query_text=query, max_results=10, include_text=False
    )
    return json.dumps([h.__dict__ for h in hits])

@tool("fetch_file")
async def tool_fetch_file(filename_or_path: str) -> str:
    """Fetch a file from the repository.

    Args:
        filename_or_path: The full or partial path to the file to fetch.

    Returns:
        JSON string containing the file contents or metadata.
    """
    client = GitHubClient()
    item = await client.fetch_file(TARGET_REPO, filename_or_path, include_text=True)
    return json.dumps(item.__dict__ if item else {})

async def stale_issues_node(state: State, config: RunnableConfig) -> dict[str, Any]:
    """Select the oldest stale issue, then let an agent investigate using tools."""
    model = _build_llm()
    # Agent with investigation tools + ProposalResponse for structured output
    # Limit tool calls to 3 per tool to avoid recursion errors
    agent = create_react_agent(
        model,
        [tool_search_issues, tool_search_code, tool_fetch_file, ProposalResponse]
    )

    client = GitHubClient()
    # Fetch a larger set and pick the oldest (least recently updated)
    stale_issues = await client.find_stale_open_issues(TARGET_REPO)
    if not stale_issues:
        raise RuntimeError(f"No stale issues found in {TARGET_REPO}.")

    #def _parse_ts(s: str) -> _dt.datetime:
    #   # Normalize Z suffix to +00:00 for fromisoformat
    #    return _dt.datetime.fromisoformat(s.replace("Z", "+00:00"))

    #oldest = sorted(stale_issues, key=lambda it: _parse_ts(it.updated_at))[0]
    oldest = stale_issues[0]
    # Fetch full comments for the selected issue (all pages)
    all_comments: list[dict[str, Any]] = await client.get_issue_comments(
        TARGET_REPO, oldest.number
    )

    # Provide the issue context and instruct to use tools to investigate
    prompt = (
        "You are a helpful GitHub maintainer assistant.\n"
        f"Repository: {TARGET_REPO}\n\n"
        "Investigate whether the following stale issue is obsolete and can be closed.\n"
        "You have the full issue details below (including all comments).\n"
        "Use the available tools `search_issues` and `search_code` to look for related discussions and code changes.\n"
        "Use the tool `fetch_file` to retrieve specific files if needed.\n"
        "You may call each tool at most 3 times per issue investigation.\n"
        "Then call the ProposalResponse tool exactly once with your decision.\n\n"
        "Guidance:\n"
        "- Prefer closing issues that lack reproduction details and seem environment-specific.\n"
        "- Avoid closing issues with recent activity or clear ongoing work.\n"
        "- Do NOT close an issue that has a comment posted after the 'stale' comment.\n"
        "- If closure seems reasonable, draft an empathetic comment referencing relevant findings.\n"
        "- If insufficient information, set can_close=false with rationale.\n\n"
        "IMPORTANT: When using the `search_code` tool, do NOT use boolean operators (AND, OR, NOT) in your query.\n"
        "Use simple keyword or phrase queries only. For example:\n"
        "- 'form recognizer'\n"
        "- 'Document Intelligence'\n"
        "- 'documentanalysis'\n"
        "- 'DocumentAnalysisClient'\n"
        "If you want to search for multiple keywords, run separate queries for each keyword or phrase.\n\n"
        "Issue details:\n"
        f"- number: {oldest.number}\n"
        f"- title: {oldest.title}\n"
        f"- url: {oldest.url}\n"
        f"- updatedAt: {oldest.updated_at}\n"
        f"- labels: {', '.join(oldest.labels) if oldest.labels else '(none)'}\n"
        f"- body:\n{(oldest.body[:4000])}\n"
        f"- comments ({len(all_comments)} total):\n{json.dumps(all_comments, ensure_ascii=False)}\n"
    )

    result = await agent.ainvoke({"messages": prompt})
    messages = result.get("messages") or []

    # Find the last AIMessage and its ProposalResponse tool call
    observed_tool_calls: list[str] = []
    last_ai_text: str | None = None
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            if isinstance(m.content, str):
                last_ai_text = last_ai_text or m.content
            for call in (m.tool_calls or []):
                if call.get("name") == "ProposalResponse":
                    args = call.get("args")
                    if isinstance(args, dict):
                        # normalize into single proposal dict
                        return {"proposal": {
                            "number": args.get("number"),
                            "title": args.get("title"),
                            "url": args.get("url"),
                            "can_close": args.get("can_close"),
                            "rationale": args.get("rationale"),
                            "suggested_comment": args.get("suggested_comment"),
                        }}
                name = call.get("name")
                if isinstance(name, str):
                    observed_tool_calls.append(name)
    # If we didn't find the tool call, fail fast (no fallbacks)
    if observed_tool_calls:
        logger.info("[stale_issues_node] Tool calls observed: %s", observed_tool_calls)
    if last_ai_text:
        snippet = (last_ai_text[:400] + "…") if len(last_ai_text) > 400 else last_ai_text
        logger.info("[stale_issues_node] Last AI text: %s", snippet)
    raise RuntimeError(
        f"ProposalResponse tool was not called. Observed tool calls: {observed_tool_calls or 'none'}."
    )


async def review_issue_node(state: State, config: RunnableConfig) -> dict[str, Any]:
    """Interrupt for the single issue to approve/edit/skip posting a close comment."""
    assert state.proposal is not None, "Proposal must be parsed before review."
    issue = state.proposal
    number = issue.get("number")
    title = issue.get("title")
    url = issue.get("url")
    suggested_comment = issue.get("suggested_comment")
    can_close = issue.get("can_close")

    description = (
        f"# Review Issue #{number}: {title}\n\n"
        f"URL: {url}\n\n"
        "If you approve, you're confirming this issue can be closed and the comment can be posted.\n\n"
        "- Accept: approve as-is.\n"
        "- Edit: update the suggested_comment and/or can_close.\n"
        "- Respond: leave a note (no changes).\n"
        "- Ignore: skip this issue.\n"
    )

    action_request = ActionRequest(
        action=f"Review issue closure: {title}",
        args={
            "number": number,
            "title": title,
            "url": url,
            "can_close": can_close,
            "suggested_comment": suggested_comment,
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
        "comment": None,
        "note": None,
    }

    rtype = human_response.get("type")
    if rtype in ("accept", "edit"):
        # Expect an ActionRequest in args, with edited args under 'args'
        ar_raw: Any = human_response.get("args")
        if isinstance(ar_raw, dict):
            inner = ar_raw.get("args")
            ar_args: dict[str, Any] = inner if isinstance(inner, dict) else {}
        else:
            ar_args = {}
        # Values may be strings; coerce can_close
        updated_can_close = ar_args.get("can_close", can_close)
        if isinstance(updated_can_close, str):
            updated_can_close = updated_can_close.lower() in ("true", "1", "yes")
        updated_comment = ar_args.get("suggested_comment", suggested_comment)
        decision.update({
            "approved": bool(updated_can_close),
            "comment": updated_comment,
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
    """If approved, post the comment and close the issue via GraphQL. Otherwise do nothing."""
    assert state.proposal is not None and state.decision is not None
    issue = state.proposal
    decision = state.decision
    number = int(issue.get("number"))
    approved = bool(decision.get("approved"))
    comment = (decision.get("comment") or "").strip()

    if not approved:
        return {}

    client = GitHubClient()
    # Ensure a non-empty comment; if empty, supply a generic one
    close_comment = comment or (
        "Closing as stale. If you're still affected, please open a new issue with updated details."
    )
    await client.close_issue_with_comment(TARGET_REPO, number, close_comment)
    return {}


TARGET_REPO = os.getenv("TARGET_REPO", "Azure-samples/azure-search-openai-demo")
# Define a new graph
workflow = StateGraph(State)

# Add nodes (single linear flow)
workflow.add_node("stale_issues", stale_issues_node)
workflow.add_node("review_issue", review_issue_node)
workflow.add_node("apply_decision", apply_decision_node)
workflow.add_node("finalize", finalize_node)

# Linear edges
workflow.add_edge("__start__", "stale_issues")
workflow.add_edge("stale_issues", "review_issue")
workflow.add_edge("review_issue", "apply_decision")
workflow.add_edge("apply_decision", "finalize")

# Compile the workflow into an executable graph
graph = workflow.compile()
graph.name = "Single Issue Closure"  # Custom name in LangSmith
