# Issue Triager Agent

[![Open in GitHub Codespaces](https://img.shields.io/static/v1?style=for-the-badge&label=GitHub+Codespaces&message=Open&color=brightgreen&logo=github)](https://codespaces.new/top-ace-friend/stale-issue-closer-agent)
[![Open in Dev Containers](https://img.shields.io/static/v1?style=for-the-badge&label=Dev%20Containers&message=Open&color=blue&logo=visualstudiocode)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/top-ace-friend/stale-issue-closer-agent)

This repository provides a LangGraph-based agentic system that proposes closing stale GitHub issues, with a human-in-the-loop review step using Agent Inbox. It selects stale issues from a target repository, investigates with repository-aware tools, and then interrupts for you to approve, edit, respond, or ignore before it posts a closing comment and closes the issue. The agent uses Azure OpenAI for the LLM.

- Graph ID: `agent` (see `langgraph.json`)
- Default target repo: `Azure-samples/azure-search-openai-demo` (configurable via `TARGET_REPO`)

Contents

- Getting started
  - GitHub Codespaces
  - VS Code Dev Containers
  - Local environment
- GitHub authentication (required)
- Configuring Azure AI models
- Running the stale issue closer
- Agent Inbox setup
- Cost estimate
- Developer tasks
- Resources

## Getting started

1. Make sure the following are installed:

    - Python 3.11+
    - Git
    - [uv](https://docs.astral.sh/uv/) (for dependency management)

2. Clone the repository:

    ```bash
    git clone https://github.com/top-ace-friend/stale-issue-closer-agent
    cd stale-issue-closer-agent
    ```

3. Create the virtual environment and install dependencies (this will create `.venv` and `uv.lock`):

    ```bash
    uv sync
    ```

## Configuring Azure AI models

This project uses Azure OpenAI for the LLM. This repository includes IaC (Bicep) to provision an Azure OpenAI deployment and writes a ready-to-use `.env` file.

1. Install the [Azure Developer CLI (azd)](https://aka.ms/install-azd)

2. Sign in to Azure:

    ```bash
    azd auth login
    ```

3. Provision Azure resources (this deploys Azure OpenAI and writes `.env` via a post-provision hook):

    ```bash
    azd provision
    ```

    After provisioning, your `.env` will include values like `AZURE_TENANT_ID`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_CHAT_DEPLOYMENT`, `AZURE_OPENAI_VERSION`, `AZURE_OPENAI_CHAT_MODEL`

## Configuring GitHub authentication

This project requires a GitHub personal access token to call the GitHub GraphQL and REST APIs (for searching issues/code and closing issues). It does *not* use GitHub Models for the LLM calls.

1. In GitHub Developer Settings, create a personal access token with `repo` scope.

2. Set `GITHUB_TOKEN` in your shell or in the `.env` file:

    ```bash
    export GITHUB_TOKEN=your_personal_access_token
    ```

3. Set the target repository (optional, defaults to `Azure-samples/azure-search-openai-demo`):

    ```bash
    # Full name, e.g., owner/name. Default is Azure-samples/azure-search-openai-demo
    export TARGET_REPO=owner/name
    ```

## Configuring Langsmith

This project uses Langsmith for tracing and for integration with Agent Inbox. To enable Langsmith tracing, set the following environment variables in your shell or in the `.env` file:

```bash
LANGSMITH_TRACING="true"
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY="<your_langsmith_api_key>"
LANGSMITH_PROJECT="<your_project_name>"
```

## Running the triager

This project uses `uv` and LangGraph’s dev server.

1. Ensure dependencies are installed:

    ```bash
    uv sync
    ```

2. Start the LangGraph dev server (inside the uv-managed virtualenv):

    ```bash
    uvx --from "langgraph-cli[inmem]" --with-editable . langgraph dev --allow-blocking
    ```

   When it’s running, a browser tab should open to LangGraph Studio via LangSmith. If it doesn’t, open this URL: <https://smith.langchain.com/studio/thread?baseUrl=http%3A%2F%2F127.0.0.1%3A2024>

3. In LangGraph Studio, start a new run of the `agent` graph. The agent will select a stale issue from `TARGET_REPO`, investigate with tools, then interrupt for review.

## Running the Agent Inbox

This repository includes the [Langchain Agent Inbox UI](https://github.com/langchain-ai/agent-inbox) as a git submodule at `agent-inbox` (tracking the `ui-improve` branch of <https://github.com/langchain-ai/agent-inbox>). Use it to review and act on issue proposals produced by the triager.

1. If you cloned this repo without `--recurse-submodules`, initialize/update the submodule first:

    ```bash
    git submodule update --init --recursive
    ```

2. Start the local Agent Inbox dev server:

    ```bash
    cd agent-inbox
    yarn install   # first time only (or when dependencies change)
    yarn dev
    ```

3. Navigate to the server running at http://localhost:3000

4. Configure the inbox:

    - Add your LangSmith API key: Click the "Settings" button in the sidebar, and enter your LangSmith API key.
    - Graph/Assistant ID: `agent` (matches `langgraph.json`)
    - Deployment URL: `http://127.0.0.1:2024`

5. When the triager makes a proposal, it should show up as a thread in the inbox. Accept or edit to let the triager continue; if approved it will apply the actions (comment / labels / close).


## Cost estimate

LLM usage varies per issue depending on the number of tool calls and the length of the issue and comments. With Azure OpenAI, cost depends on the model and tokens processed. See pricing: <https://azure.microsoft.com/pricing/details/cognitive-services/openai-service/>

## Developer tasks

Common dev tasks are available via `uv` and the included Makefile targets:

- Run tests:

  ```bash
  uv run -- python -m pytest
  ```

- Lint / format:

  ```bash
  uv run -- ruff check .
  uv run -- ruff format .
  ```

- Type checking:

  ```bash
  uv run -- mypy src
  ```

## Resources

- Agent Inbox: <https://github.com/langchain-ai/agent-inbox>
- LangGraph docs: <https://langchain-ai.github.io/langgraph/>
- LangGraph Studio (via LangSmith): <https://smith.langchain.com/>
- LangGraph Studio (via LangSmith): <https://smith.langchain.com/>
