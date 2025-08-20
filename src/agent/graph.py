"""LangGraph graph that uses the GitHub MCP server to propose closing stale issues.

With a human-in-the-loop review step using Agent Inbox-compatible interrupts.

Environment variables used:
- GITHUB_TOKEN: Required for MCP GitHub tools. Token with access to GitHub Models/MCP server.
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
from typing import Any, cast

import azure.identity.aio
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_mcp_adapters.client import (  # type: ignore[import-untyped]
    MultiServerMCPClient,
)
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
        azure_tenant_id = os.environ["AZURE_TENANT_ID"]

        token_provider = azure.identity.aio.get_bearer_token_provider(
            azure.identity.aio.AzureDeveloperCliCredential(tenant_id=azure_tenant_id),
            "https://cognitiveservices.azure.com/.default",
        )
        return AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=azure_version,
            azure_ad_async_token_provider=token_provider,
        )
    # Configure the LLM via GitHub Models endpoint
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise RuntimeError(
            "GITHUB_TOKEN is required. Set it to a token that can access GitHub Models and the MCP GitHub server."
        )
    return ChatOpenAI(
        model=os.getenv("GITHUB_MODEL", "gpt-4o"),
        base_url="https://models.inference.ai.azure.com",
        api_key=SecretStr(token),
    )


async def _get_mcp_tools(allow: set[str] | None = None, require: bool = False) -> list[Any]:
    """Fetch MCP tools from the GitHub server.

    - allow: if provided, only return tools whose name is in this set.
    - require: when True, raise if any tool in `allow` is missing.
    """
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise RuntimeError(
            "GITHUB_TOKEN is required. Set it to a token that can access GitHub Models and the MCP GitHub server."
        )
    client = MultiServerMCPClient(
        {
            "github": {
                "url": "https://api.githubcopilot.com/mcp/",
                "transport": "streamable_http",
                "headers": {
                    "Authorization": f"Bearer {token}",
                },
            }
        }
    )
    all_tools = cast(list[Any], await client.get_tools())
    if allow is None:
        return all_tools

    available_names = {
        name for name in (getattr(t, "name", None) for t in all_tools) if isinstance(name, str)
    }
    missing = {n for n in allow if n not in available_names}
    if require and missing:
        raise RuntimeError(f"Missing required MCP tools: {sorted(missing)}")
    return [t for t in all_tools if getattr(t, "name", None) in allow]


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
    return json.dumps(payload)


async def stale_issues_node(state: State, config: RunnableConfig) -> dict[str, Any]:
    """Use the GitHub MCP server tools to find stale issues and propose closures.

    Returns a dict that updates the state with a JSON string in `proposals_json`.
    """
    # Inline reader agent creation (read-only tools + ProposalResponse)
    model = _build_llm()
    filtered_tools = await _get_mcp_tools(
        allow={"search_issues", "get_issue", "search_code"}, require=True
    )
    filtered_tools.append(ProposalResponse)
    agent = create_react_agent(model, filtered_tools)

    # Use preloaded template (read once at import time)
    template = STALE_PROMPT_TEMPLATE

    prompt = template.replace("{{TARGET_REPO}}", TARGET_REPO)

    # Invoke once; require ProposalResponse tool call and capture its args
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
    """If approved, post the comment and close the issue. Otherwise do nothing."""
    assert state.proposal is not None and state.decision is not None
    issue = state.proposal
    decision = state.decision
    number = issue.get("number")
    approved = bool(decision.get("approved"))
    comment = decision.get("comment") or ""

    # Inline writer agent creation (write-only tools)
    model = _build_llm()
    filtered_tools = await _get_mcp_tools(
        allow={"update_issue", "add_issue_comment"}, require=True
    )
    agent = create_react_agent(model, filtered_tools)

    if not approved:
        return {}

    instructions = (
        f"For repo {TARGET_REPO}, for issue #{number}:\n"
        "1) Post the following comment verbatim.\n"
        "2) Close the issue.\n\n"
        f"Comment:\n{comment}\n\n"
        "Use only GitHub MCP tools to perform these actions. Then reply 'ok'."
    )

    async for _ in agent.astream_events({"messages": instructions}, version="v2"):
        pass
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
