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

You have a few options for getting set up. The quickest is GitHub Codespaces, but you can also use a Dev Container locally or your native environment.

### GitHub Codespaces

Run this repo virtually in your browser:

1. Open the repository in Codespaces (it may take a few minutes to build):

	[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/top-ace-friend/stale-issue-closer-agent)

2. Open a terminal in the Codespace.
3. Continue with the steps in “Running the stale issue closer”.

### VS Code Dev Containers

Open the project in a local Dev Container using the Dev Containers extension:

1. Start Docker Desktop.
2. Open the project in a Dev Container:

	[![Open in Dev Containers](https://img.shields.io/static/v1?style=for-the-badge&label=Dev%20Containers&message=Open&color=blue&logo=visualstudiocode)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/top-ace-friend/stale-issue-closer-agent)

3. Once the container is ready, open a terminal.
4. Continue with the steps in “Running the stale issue closer”.

### Local environment

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

## GitHub authentication (required)

This project requires a GitHub personal access token to call the GitHub GraphQL and REST APIs (for searching issues/code and closing issues). It does not use GitHub Models for the LLM.

- In GitHub Codespaces, `GITHUB_TOKEN` may already be available.
- Locally or in Dev Containers, set it in your shell or in a local `.env` file:

  ```bash
  # Shell example (zsh)
  export GITHUB_TOKEN=your_personal_access_token
  ```

## Configuring Azure AI models

Azure OpenAI is required for the LLM. This repository includes IaC (Bicep) to provision an Azure OpenAI deployment and writes a ready-to-use `.env` file.

1. Install the Azure Developer CLI (azd):

	See the install guide: <https://aka.ms/install-azd>

2. Sign in to Azure:

	```bash
	azd auth login
	# If that fails (e.g., in Codespaces), try device code:
	azd auth login --use-device-code
	```

3. Provision Azure resources (this deploys Azure OpenAI and writes `.env` via a post-provision hook):

	```bash
	azd provision
	```

	After provisioning, your `.env` will include values like:

	- `API_HOST=azure`
	- `AZURE_TENANT_ID`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_CHAT_DEPLOYMENT`, `AZURE_OPENAI_VERSION`, `AZURE_OPENAI_CHAT_MODEL`

4. Cleanup when finished:

	```bash
	azd down
	```

Important:

- You still need `GITHUB_TOKEN` for the GitHub GraphQL and REST APIs used by this project.

## Running the stale issue closer

This project uses `uv` and LangGraph’s dev server.

1. Ensure dependencies are installed:

	```bash
	uv sync
	```

2. (Optional) Set the target repository and other environment variables:

	```bash
	# Full name, e.g., owner/name. Default is Azure-samples/azure-search-openai-demo
	export TARGET_REPO=owner/name
	# GitHub token is required in all modes
	export GITHUB_TOKEN=your_personal_access_token
	```

3. Start the LangGraph dev server (runs inside the uv-managed virtualenv):

	```bash
	uvx --from "langgraph-cli[inmem]" --with-editable . langgraph dev --allow-blocking
	```

	When it’s running, a browser tab should open to LangGraph Studio via LangSmith. If it doesn’t, open this URL: <https://smith.langchain.com/studio/thread?baseUrl=http%3A%2F%2F127.0.0.1%3A2024>

4. In LangGraph Studio, start a new run of the `agent` graph. The agent will select a stale issue from `TARGET_REPO`, investigate with tools, then interrupt for review.

## Agent Inbox setup

Use Agent Inbox to review and act on the interruption created by the graph.

1. Visit <https://dev.agentinbox.ai>
2. Add your graph with these fields:
	- Graph/Assistant ID: `agent` (matches `langgraph.json`)
	- Deployment URL: `http://127.0.0.1:2024`
	- Name: any display name (optional)
3. Open the `agent` entry from the sidebar and refresh if needed. You should see the interrupted item. Click it to review and choose an action:
	- Accept: approve as-is
	- Edit: update the suggested comment and/or can_close
	- Respond: leave a note (no changes applied)
	- Ignore: skip the issue

After you accept or edit, the graph resumes and, if approved, posts a comment and closes the issue.

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

