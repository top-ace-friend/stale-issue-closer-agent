"""GitHub client for repo maintenance tasks (GraphQL + selected REST endpoints).

Functions provided:
- find_stale_open_issues
- close_issue_with_comment
- search_issues_with_bodies
- search_codebase (REST /search/code)
- fetch_file (REST /search/code + contents)
- list_repository_files (REST git trees API)

Authentication:
- Uses a GitHub token with repo read/write and search scopes.
- Reads from env var GITHUB_TOKEN by default; can be passed explicitly.

Notes:
- Primarily uses GraphQL v4; some helper methods use REST endpoints where GraphQL lacks feature parity (code search, file fetch, tree listing).
- All methods are async and use httpx.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import httpx

GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"


@dataclass
class IssueItem:
    """Lightweight issue record returned by searches.

    Attributes:
        number: Issue number.
        title: Issue title.
        url: Web URL to the issue.
        state: Current issue state (e.g., OPEN, CLOSED).
        updated_at: ISO timestamp of last update.
        created_at: ISO timestamp of creation.
        author: Login of issue author, if available.
        body: Markdown body of the issue.
        id: GraphQL node ID of the issue.
        labels: List of label names applied to the issue.
    """

    number: int
    title: str
    url: str
    state: str
    updated_at: str
    created_at: str
    author: str | None
    body: str
    id: str  # node id
    labels: List[str]


@dataclass
class CodeSearchItem:
    """A single file hit from code search.

    Attributes:
        repository: Full name (owner/name) of the repository.
        path: Path to the file within the repository.
        text: File contents when requested and not binary.
        is_binary: Whether the blob is binary.
        byte_size: Size of the blob in bytes when known.
    """

    repository: str  # owner/name
    path: str
    text: str | None
    is_binary: bool
    byte_size: int | None
    snippet: str | None = None


class GitHubClient:
    """Async GitHub GraphQL client focused on issues and code search."""

    def __init__(self, token: str | None = None, timeout: float = 30.0) -> None:
        """Initialize the client with a GitHub token and timeout.

        Args:
            token: Personal access token to authenticate requests. Falls back to env var GITHUB_TOKEN.
            timeout: Request timeout in seconds.
        """
        self.token = token or os.getenv("GITHUB_TOKEN")
        if not self.token:
            raise RuntimeError("GITHUB_TOKEN is required for GitHubClient")
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
        }
        self.timeout = httpx.Timeout(timeout)
        self._viewer_login: str | None = None  # cache for authenticated user login
        self._repo_labels_cache: dict[str, list[Dict[str, Any]]] = {}

    async def get_repository_labels(self, repo: str, force_refresh: bool = False) -> list[Dict[str, Any]]:
        """Return cached repository labels (list of dicts) for repo.

        Caches per-repo list on first fetch. Set force_refresh=True to refetch.
        """
        if not force_refresh and repo in self._repo_labels_cache:
            return self._repo_labels_cache[repo]
        labels = await self.list_repository_labels(repo)
        self._repo_labels_cache[repo] = labels
        return labels

    async def get_viewer_login(self) -> str:
        """Return the login of the authenticated user (GitHub token owner).

        Uses a lightweight GraphQL query and caches the result for subsequent calls.
        Falls back to 'maintainer' if the query unexpectedly fails.
        """
        if self._viewer_login:
            return self._viewer_login
        query = """
        query { viewer { login } }
        """
        try:
            data = await self._graphql(query, {})
            viewer = data.get("viewer") if isinstance(data, dict) else None
            login = (viewer or {}).get("login") or "maintainer"
            self._viewer_login = login
            return login
        except Exception:  # noqa: BLE001
            return "maintainer"

    async def _graphql(self, query: str, variables: Dict[str, Any]) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                GITHUB_GRAPHQL_URL,
                headers=self.headers,
                json={"query": query, "variables": variables},
            )
            resp.raise_for_status()
            data = resp.json()
        if "errors" in data and data["errors"]:
            raise RuntimeError(f"GitHub GraphQL error: {data['errors']}")
        return data["data"]

    @staticmethod
    def _split_repo(full_name: str) -> tuple[str, str]:
        if "/" not in full_name:
            raise ValueError("repo must be in the form 'owner/name'")
        owner, name = full_name.split("/", 1)
        return owner, name

    async def find_stale_open_issues(
        self,
        repo: str,
        max_results: int = 5,
    ) -> List[IssueItem]:
        """Find open stale issues, sorted by last update date."""
        owner, name = self._split_repo(repo)
        q = f"repo:{owner}/{name} is:issue is:open sort:updated-desc label:Stale"
        query = """
        query($q: String!, $first: Int!, $after: String) {
          search(query: $q, type: ISSUE, first: $first, after: $after) {
            issueCount
            pageInfo { hasNextPage endCursor }
            nodes {
              __typename
              ... on Issue {
                id
                number
                title
                url
                state
                updatedAt
                createdAt
                author { login }
                labels(first: 20) { nodes { name } }
                body
              }
            }
          }
        }
        """

        results: List[IssueItem] = []
        after: str | None = None
        page_size = min(50, max_results)
        while True:
            data = await self._graphql(
                query, {"q": q, "first": page_size, "after": after}
            )
            search = data["search"]
            for node in search["nodes"]:
                if node.get("__typename") != "Issue":
                    continue
                labels = [
                    n["name"] for n in (node.get("labels", {}).get("nodes", []) or [])
                ]
                item = IssueItem(
                    id=node["id"],
                    number=node["number"],
                    title=node["title"],
                    url=node["url"],
                    state=node["state"],
                    updated_at=node["updatedAt"],
                    created_at=node["createdAt"],
                    author=(node.get("author") or {}).get("login"),
                    body=node.get("body") or "",
                    labels=labels,
                )
                results.append(item)
                if len(results) >= max_results:
                    return results
            if not search["pageInfo"]["hasNextPage"]:
                break
            after = search["pageInfo"]["endCursor"]
        return results

    async def close_issue_with_comment(
        self, repo: str, issue_number: int, comment: str
    ) -> Dict[str, Any]:
        """Post a comment on an issue then close it."""
        owner, name = self._split_repo(repo)
        get_issue_q = """
        query($owner: String!, $name: String!, $number: Int!) {
          repository(owner: $owner, name: $name) {
            issue(number: $number) { id url number title state }
          }
        }
        """
        data = await self._graphql(
            get_issue_q, {"owner": owner, "name": name, "number": issue_number}
        )
        issue = data["repository"]["issue"]
        if not issue:
            raise RuntimeError(f"Issue #{issue_number} not found in {repo}")
        issue_id = issue["id"]

        add_comment_mut = """
        mutation($input: AddCommentInput!) {
          addComment(input: $input) { clientMutationId }
        }
        """
        await self._graphql(
            add_comment_mut, {"input": {"subjectId": issue_id, "body": comment}}
        )

        close_issue_mut = """
        mutation($input: CloseIssueInput!) {
          closeIssue(input: $input) { issue { number state } }
        }
        """
        data2 = await self._graphql(close_issue_mut, {"input": {"issueId": issue_id}})
        closed = data2["closeIssue"]["issue"]
        return {
            "number": closed["number"],
            "state": closed["state"],
            "url": issue["url"],
        }

    async def post_comment(self, repo: str, issue_number: int, body: str) -> None:
        """Post a comment on an issue without changing its state."""
        owner, name = self._split_repo(repo)
        get_issue_q = """
            query($owner: String!, $name: String!, $number: Int!) {
                repository(owner: $owner, name: $name) { issue(number: $number) { id } }
            }
            """
        data = await self._graphql(
            get_issue_q, {"owner": owner, "name": name, "number": issue_number}
        )
        issue = data["repository"]["issue"]
        if not issue:
            raise RuntimeError(f"Issue #{issue_number} not found in {repo}")
        issue_id = issue["id"]
        add_comment_mut = """
            mutation($input: AddCommentInput!) { addComment(input: $input) { clientMutationId } }
            """
        await self._graphql(
            add_comment_mut, {"input": {"subjectId": issue_id, "body": body}}
        )

    async def close_issue(self, repo: str, issue_number: int) -> None:
        """Close an issue without commenting."""
        owner, name = self._split_repo(repo)
        get_issue_q = """
            query($owner: String!, $name: String!, $number: Int!) {
                repository(owner: $owner, name: $name) { issue(number: $number) { id } }
            }
            """
        data = await self._graphql(
            get_issue_q, {"owner": owner, "name": name, "number": issue_number}
        )
        issue = data["repository"]["issue"]
        if not issue:
            raise RuntimeError(f"Issue #{issue_number} not found in {repo}")
        issue_id = issue["id"]
        close_issue_mut = """
            mutation($input: CloseIssueInput!) { closeIssue(input: $input) { issue { number state } } }
            """
        await self._graphql(close_issue_mut, {"input": {"issueId": issue_id}})

    async def assign_issue_to_copilot(self, repo: str, issue_number: int, copilot_login: str = "copilot-swe-agent") -> dict[str, Any]:
        """Assign an existing issue to the GitHub Copilot agent bot.

        This uses the GraphQL API to:
        1. Fetch the issue node ID and repository suggested actors (CAN_BE_ASSIGNED).
        2. Locate the Copilot bot's node ID by login (default: "copilot-swe-agent").
        3. Call addAssigneesToAssignable to assign the issue to Copilot.

        Args:
            repo: Full repository name ("owner/name").
            issue_number: Issue number to assign.
            copilot_login: Login of the Copilot bot (override if different in future).

        Returns:
            Dict with keys: number, url, assigned (bool), assignee_login.

        Raises:
            RuntimeError: If the issue or Copilot bot cannot be found / assignment fails.
        """
        owner, name = self._split_repo(repo)
        query = """
        query($owner: String!, $name: String!, $number: Int!) {
          repository(owner: $owner, name: $name) {
            id
            suggestedActors(capabilities: [CAN_BE_ASSIGNED], first: 100) {
              nodes { login __typename ... on Bot { id } ... on User { id } }
            }
            issue(number: $number) { id number url }
          }
        }
        """
        data = await self._graphql(query, {"owner": owner, "name": name, "number": issue_number})
        repo_node = data.get("repository") if isinstance(data, dict) else None
        if not repo_node:
            raise RuntimeError(f"Repository {repo} not found")
        issue = (repo_node.get("issue") if isinstance(repo_node, dict) else None) or None
        if not issue:
            raise RuntimeError(f"Issue #{issue_number} not found in {repo}")
        issue_id = issue.get("id")
        suggested = ((repo_node.get("suggestedActors") or {}).get("nodes") or []) if isinstance(repo_node, dict) else []
        copilot_node = next((n for n in suggested if n.get("login") == copilot_login and n.get("id")), None)

        # If Copilot is not in suggested assignable actors, do a lightweight REST attempt before giving up.
        if not copilot_node:
            logging.warning(
                "Copilot '%s' not present in suggested assignable actors for %s; attempting REST fallback.",
                copilot_login,
                repo,
            )
            return await self._assign_issue_via_rest(repo, issue_number, copilot_login, issue.get("url"))

        # If the node is a Bot, GitHub currently rejects addAssigneesToAssignable with NOT_FOUND (expects User IDs).
        typename = copilot_node.get("__typename")
        if typename == "Bot":
            logging.info(
                "Suggested actor '%s' is a Bot; attempting REST fallback instead of GraphQL assignment.",
                copilot_login,
            )
            return await self._assign_issue_via_rest(repo, issue_number, copilot_login, issue.get("url"))

        mutation = """
        mutation($assignableId: ID!, $assigneeIds: [ID!]!) {
          addAssigneesToAssignable(input: {assignableId: $assignableId, assigneeIds: $assigneeIds}) {
            assignable { ... on Issue { number url } }
          }
        }
        """
        try:
            add_res = await self._graphql(
                mutation,
                {"assignableId": issue_id, "assigneeIds": [copilot_node["id"]]},
            )
        except Exception as exc:  # noqa: BLE001
            # Specific handling for NOT_FOUND referencing BOT_ ids – fallback to REST
            msg = str(exc)
            if "BOT_" in msg and "NOT_FOUND" in msg:
                logging.warning(
                    "GraphQL assignment failed due to bot ID (NOT_FOUND). Falling back to REST: %s",
                    msg,
                )
                return await self._assign_issue_via_rest(repo, issue_number, copilot_login, issue.get("url"))
            logging.error("Failed assigning Copilot to #%s in %s: %s", issue_number, repo, exc)
            return {"number": issue.get("number"), "url": issue.get("url"), "assigned": False, "assignee_login": copilot_login, "reason": str(exc)}
        assignable = ((add_res.get("addAssigneesToAssignable") or {}).get("assignable") if isinstance(add_res, dict) else None) or {}
        number = assignable.get("number", issue.get("number"))
        url = assignable.get("url", issue.get("url"))
        return {"number": number, "url": url, "assigned": True, "assignee_login": copilot_login}

    async def _assign_issue_via_rest(self, repo: str, issue_number: int, login: str, issue_url: str | None) -> dict[str, Any]:
        """Fallback assignment using REST Issues API.

        GitHub currently may not allow assigning certain bot accounts via GraphQL (expects a User node ID),
        so we try the classic REST endpoint. If this also fails, we return a non-fatal result.
        """
        owner, name = self._split_repo(repo)
        url = f"https://api.github.com/repos/{owner}/{name}/issues/{issue_number}/assignees"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.post(url, headers=self.headers, json={"assignees": [login]})
                if resp.status_code in (201, 200):
                    return {"number": issue_number, "url": issue_url, "assigned": True, "assignee_login": login, "via": "rest"}
                # If unprocessable or not found, treat as not assignable.
                logging.warning(
                    "REST assignment of %s to #%s in %s failed (%s): %s",
                    login,
                    issue_number,
                    repo,
                    resp.status_code,
                    resp.text,
                )
                return {"number": issue_number, "url": issue_url, "assigned": False, "assignee_login": login, "via": "rest", "reason": f"HTTP {resp.status_code}"}
            except Exception as exc:  # noqa: BLE001
                logging.warning(
                    "REST assignment exception for %s on #%s in %s: %s",
                    login,
                    issue_number,
                    repo,
                    exc,
                )
                return {"number": issue_number, "url": issue_url, "assigned": False, "assignee_login": login, "via": "rest", "reason": str(exc)}

    async def add_label(self, repo: str, issue_number: int, label: str) -> None:
        """Add a label to an issue (creates if repository already has it)."""
        owner, name = self._split_repo(repo)
        # Fetch label id or create it
        label_query = """
            query($owner: String!, $name: String!, $label: String!, $number: Int!) {
                repository(owner: $owner, name: $name) {
                    label(name: $label) { id name }
                    issue(number: $number) { id }
                }
            }
            """
        data = await self._graphql(
            label_query,
            {"owner": owner, "name": name, "label": label, "number": issue_number},
        )
        repo_node = data["repository"]
        issue = repo_node.get("issue")
        if not issue:
            raise RuntimeError(f"Issue #{issue_number} not found in {repo}")
        issue_id = issue["id"]
        label_node = repo_node.get("label")
        if not label_node:
            # Create label
            create_label_mut = """
                    mutation($input: CreateLabelInput!) { createLabel(input: $input) { label { id name } } }
                    """
            color = "ededed"
            create_res = await self._graphql(
                create_label_mut,
                {
                    "input": {
                        "repositoryId": repo_node.get("id"),
                        "name": label,
                        "color": color,
                    }
                },
            )
            label_id = create_res["createLabel"]["label"]["id"]
        else:
            label_id = label_node["id"]
        add_labels_mut = """
            mutation($input: AddLabelsToLabelableInput!) { addLabelsToLabelable(input: $input) { clientMutationId } }
            """
        await self._graphql(
            add_labels_mut, {"input": {"labelIds": [label_id], "labelableId": issue_id}}
        )

    async def remove_label(self, repo: str, issue_number: int, label: str) -> None:
        """Remove a label from an issue if present."""
        owner, name = self._split_repo(repo)
        label_query = """
            query($owner: String!, $name: String!, $label: String!, $number: Int!) {
                repository(owner: $owner, name: $name) {
                    label(name: $label) { id name }
                    issue(number: $number) { id labels(first: 50) { nodes { id name } } }
                }
            }
            """
        data = await self._graphql(
            label_query,
            {"owner": owner, "name": name, "label": label, "number": issue_number},
        )
        repo_node = data["repository"]
        issue = repo_node.get("issue")
        if not issue:
            raise RuntimeError(f"Issue #{issue_number} not found in {repo}")
        issue_id = issue["id"]
        labels_conn = issue.get("labels") or {}
        nodes = labels_conn.get("nodes") or []
        target = next((n for n in nodes if n.get("name") == label), None)
        if not target:
            return  # label not present
        remove_labels_mut = """
            mutation($input: RemoveLabelsFromLabelableInput!) { removeLabelsFromLabelable(input: $input) { clientMutationId } }
            """
        await self._graphql(
            remove_labels_mut,
            {"input": {"labelIds": [target["id"]], "labelableId": issue_id}},
        )

    async def get_issue(
        self,
        repo: str,
        issue_number: int,
        include_comments: bool = True,
        comments_limit: int = 50,
    ) -> Dict[str, Any] | None:
        """Fetch a single issue (and optionally recent comments) by number.

        Args:
            repo: owner/name
            issue_number: The issue number.
            include_comments: Whether to include up to `comments_limit` latest comments.
            comments_limit: Max number of comments to fetch.

        Returns:
            Dict with issue fields or None if not found.
        """
        owner, name = self._split_repo(repo)
        query_issue = """
        query($owner: String!, $name: String!, $number: Int!, $n: Int!) {
          repository(owner: $owner, name: $name) {
            issue(number: $number) {
              id number title url state updatedAt createdAt author { login }
              body
              labels(first: 20) { nodes { name } }
              comments(first: $n) {
                pageInfo { hasNextPage endCursor }
                nodes { id author { login } createdAt body }
              }
            }
          }
        }
        """
        data = await self._graphql(
            query_issue,
            {
                "owner": owner,
                "name": name,
                "number": issue_number,
                "n": max(0, comments_limit),
            },
        )
        repo_node = data.get("repository") or {}
        issue = repo_node.get("issue") if isinstance(repo_node, dict) else None
        if not issue:
            return None
        labels = [n["name"] for n in (issue.get("labels", {}).get("nodes", []) or [])]
        result: Dict[str, Any] = {
            "id": issue.get("id"),
            "number": issue.get("number"),
            "title": issue.get("title"),
            "url": issue.get("url"),
            "state": issue.get("state"),
            "updatedAt": issue.get("updatedAt"),
            "createdAt": issue.get("createdAt"),
            "author": (issue.get("author") or {}).get("login"),
            "labels": labels,
            "body": issue.get("body") or "",
        }
        if include_comments:
            comments_conn = issue.get("comments") or {}
            nodes = comments_conn.get("nodes") or []
            result["comments"] = [
                {
                    "id": c.get("id"),
                    "author": (c.get("author") or {}).get("login"),
                    "createdAt": c.get("createdAt"),
                    "body": c.get("body") or "",
                }
                for c in nodes
            ]
        return result

    async def get_issue_comments(
        self,
        repo: str,
        issue_number: int,
        max_comments: int | None = None,
        page_size: int = 50,
    ) -> List[Dict[str, Any]]:
        """Fetch all comments for an issue, with pagination.

        Args:
        repo: Full name ("owner/name").
        issue_number: Issue number.
        max_comments: Optional cap on number of comments to fetch; fetches all when None.
        page_size: Page size for each GraphQL request (default 50, max typically 100).

        Returns:
        A list of comment dicts with keys: id, author, createdAt, body.
        """
        owner, name = self._split_repo(repo)
        comments: List[Dict[str, Any]] = []
        after: str | None = None
        q = """
        query($owner: String!, $name: String!, $number: Int!, $n: Int!, $after: String) {
        repository(owner: $owner, name: $name) {
        issue(number: $number) {
            comments(first: $n, after: $after) {
            pageInfo { hasNextPage endCursor }
            nodes { id author { login } createdAt body }
            }
        }
        }
        }
        """
        while True:
            n = (
                min(page_size, max_comments - len(comments))
                if max_comments is not None
                else page_size
            )
            if n <= 0:
                break
            data = await self._graphql(
                q,
                {
                    "owner": owner,
                    "name": name,
                    "number": issue_number,
                    "n": n,
                    "after": after,
                },
            )
            repo_node = data.get("repository") or {}
            issue = repo_node.get("issue") if isinstance(repo_node, dict) else None
            if not issue:
                raise RuntimeError(f"Issue #{issue_number} not found in {repo}")
            comments_conn = issue.get("comments") or {}
            nodes = comments_conn.get("nodes") or []
            for c in nodes:
                comments.append(
                    {
                        "id": c.get("id"),
                        "author": (c.get("author") or {}).get("login"),
                        "createdAt": c.get("createdAt"),
                        "body": c.get("body") or "",
                    }
                )
                if max_comments is not None and len(comments) >= max_comments:
                    return comments
            page_info = comments_conn.get("pageInfo") or {}
            if not page_info.get("hasNextPage"):
                break
            after = page_info.get("endCursor")
        return comments

    async def search_issues_with_bodies(
        self,
        repo: str,
        query_text: str,
        max_results: int = 50,
        include_comments: bool = False,
        comments_limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Search issues and return title, body, and optional comments."""
        owner, name = self._split_repo(repo)
        q = f"repo:{owner}/{name} is:issue {query_text}".strip()
        search_q = """
        query($q: String!, $first: Int!, $after: String) {
          search(query: $q, type: ISSUE, first: $first, after: $after) {
            pageInfo { hasNextPage endCursor }
            nodes {
              __typename
              ... on Issue {
                id number title url state updatedAt createdAt author { login } body
                labels(first: 20) { nodes { name } }
              }
            }
          }
        }
        """
        results: List[Dict[str, Any]] = []
        after: str | None = None
        page_size = min(50, max_results)
        issues: List[Dict[str, Any]] = []
        while True and len(issues) < max_results:
            data = await self._graphql(
                search_q, {"q": q, "first": page_size, "after": after}
            )
            search = data["search"]
            for node in search["nodes"]:
                if node.get("__typename") != "Issue":
                    continue
                issues.append(node)
                if len(issues) >= max_results:
                    break
            if not search["pageInfo"]["hasNextPage"] or len(issues) >= max_results:
                break
            after = search["pageInfo"]["endCursor"]

        if not include_comments:
            for it in issues:
                results.append(
                    {
                        "id": it["id"],
                        "number": it["number"],
                        "title": it["title"],
                        "url": it["url"],
                        "state": it["state"],
                        "updatedAt": it["updatedAt"],
                        "createdAt": it["createdAt"],
                        "author": (it.get("author") or {}).get("login"),
                        "labels": [
                            n["name"]
                            for n in (it.get("labels", {}).get("nodes", []) or [])
                        ],
                        "body": it.get("body") or "",
                    }
                )
            return results

        for it in issues:
            comments_q = """
            query($id: ID!, $n: Int!) {
              node(id: $id) {
                ... on Issue {
                  id number url
                  comments(first: $n) {
                    totalCount
                    nodes { author { login } createdAt body }
                  }
                }
              }
            }
            """
            detail = await self._graphql(
                comments_q, {"id": it["id"], "n": comments_limit}
            )
            node = detail["node"] or {}
            comments = (node.get("comments") or {}).get("nodes") or []
            results.append(
                {
                    "id": it["id"],
                    "number": it["number"],
                    "title": it["title"],
                    "url": it["url"],
                    "state": it["state"],
                    "updatedAt": it["updatedAt"],
                    "createdAt": it["createdAt"],
                    "author": (it.get("author") or {}).get("login"),
                    "labels": [
                        n["name"] for n in (it.get("labels", {}).get("nodes", []) or [])
                    ],
                    "body": it.get("body") or "",
                    "comments": [
                        {
                            "author": (c.get("author") or {}).get("login"),
                            "createdAt": c.get("createdAt"),
                            "body": c.get("body") or "",
                        }
                        for c in comments
                    ],
                }
            )
        return results

    async def search_pull_requests(
        self,
        repo: str,
        query_text: str,
        max_results: int = 50,
    ) -> List[Dict[str, Any]]:
        """Search pull requests and return minimal metadata.

        Only the following fields are returned per PR:
            number, title, url, body (description), merged (bool), mergedAt (ISO datetime), createdAt, updatedAt.

        Args:
            repo: "owner/name" repository identifier.
            query_text: Additional search terms/qualifiers (is:pr is added automatically).
            max_results: Cap on number of pull requests returned.
        """
        owner, name = self._split_repo(repo)
        q = f"repo:{owner}/{name} is:pr {query_text}".strip()
        search_q = """
        query($q: String!, $first: Int!, $after: String) {
          search(query: $q, type: ISSUE, first: $first, after: $after) {
            pageInfo { hasNextPage endCursor }
            nodes {
              __typename
              ... on PullRequest {
                id number title url updatedAt createdAt author { login }
                body merged mergedAt
              }
            }
          }
        }
        """
        after: str | None = None
        page_size = min(50, max_results)
        pulls: List[Dict[str, Any]] = []
        while True and len(pulls) < max_results:
            data = await self._graphql(search_q, {"q": q, "first": page_size, "after": after})
            search = data["search"]
            for node in search["nodes"]:
                # The SEARCH type ISSUE returns PullRequest and Issue union; filter PRs
                if node.get("__typename") != "PullRequest":
                    continue
                pulls.append(node)
                if len(pulls) >= max_results:
                    break
            if not search["pageInfo"]["hasNextPage"] or len(pulls) >= max_results:
                break
            after = search["pageInfo"]["endCursor"]

        # Return only the minimal fields required by the caller to save tokens.
        return [
            {
                "number": pr["number"],
                "title": pr["title"],
                "url": pr["url"],  # kept for human reference
                "description": pr.get("body") or "",
                "merged": pr.get("merged"),
                "mergedAt": pr.get("mergedAt"),
            }
            for pr in pulls
        ]

    async def get_pull_request(
            self,
            repo: str,
            pr_number: int,
    ) -> Dict[str, Any] | None:
            """Fetch a single pull request by number, mirroring search_pull_requests fields.

            Returns dict with keys: number, title, url, description, merged, mergedAt
            or None if not found.
            """
            owner, name = self._split_repo(repo)
            query = """
            query($owner: String!, $name: String!, $number: Int!) {
                repository(owner: $owner, name: $name) {
                    pullRequest(number: $number) {
                        id number title url body merged mergedAt
                    }
                }
            }
            """
            data = await self._graphql(
                    query,
                    {"owner": owner, "name": name, "number": pr_number},
            )
            repo_node = data.get("repository") if isinstance(data, dict) else None
            if not repo_node:
                    return None
            pr = repo_node.get("pullRequest") if isinstance(repo_node, dict) else None
            if not pr:
                    return None
            return {
                    "number": pr.get("number"),
                    "title": pr.get("title"),
                    "url": pr.get("url"),
                    "description": pr.get("body") or "",
                    "merged": pr.get("merged"),
                    "mergedAt": pr.get("mergedAt"),
            }

    async def search_codebase(
        self,
        repo: str,
        query_text: str,
        max_results: int = 50,
        include_text: bool = True,
        ref: str | None = None,
    ) -> List[CodeSearchItem]:
        """Search for code in the repository using GitHub REST API /search/code."""
        owner, name = self._split_repo(repo)
        search_query = f"{query_text} repo:{owner}/{name}"
        url = "https://api.github.com/search/code"
        params = {"q": search_query, "per_page": min(100, max_results)}
        # Add Accept header for text-match snippets
        headers = dict(self.headers)
        headers["Accept"] = "application/vnd.github.text-match+json"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(url, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()
            results: List[CodeSearchItem] = []
            for item in data.get("items", []):
                repo_nwo = item["repository"]["full_name"]
                path = item["path"]
                is_binary = False
                byte_size = item.get("size")
                text: str | None = None
                # Extract text_matches (snippets)
                text_matches = item.get("text_matches")
                snippet = None
                if text_matches and isinstance(text_matches, list):
                    # Join all fragments for simplicity
                    snippet = "\n".join(
                        [
                            m.get("fragment", "")
                            for m in text_matches
                            if m.get("fragment")
                        ]
                    )
                if include_text:
                    # Fetch file content
                    contents_url = item["url"].replace("/search/code", "/repos")
                    file_resp = await client.get(contents_url, headers=self.headers)
                    file_resp.raise_for_status()
                    file_data = file_resp.json()
                    if file_data.get("encoding") == "base64":
                        import base64

                        try:
                            text = base64.b64decode(file_data["content"]).decode(
                                "utf-8", errors="replace"
                            )
                        except Exception:
                            is_binary = True
                            text = None
                    else:
                        text = file_data.get("content")
                results.append(
                    CodeSearchItem(
                        repository=repo_nwo,
                        path=path,
                        text=text if include_text and not is_binary else None,
                        is_binary=is_binary,
                        byte_size=byte_size,
                        snippet=snippet,
                    )
                )
                if len(results) >= max_results:
                    break
            return results

    async def fetch_file(
        self,
        repo: str,
        filename: str,
        include_text: bool = True,
        ref: str | None = None,
    ) -> CodeSearchItem | None:
        """Fetch a file by searching for its path using the GitHub code search API.

        Args:
            repo: Full name (owner/name) of the repository.
            filename: Full or partial path of the file to search for.
            include_text: Whether to fetch the file contents.
            ref: Optional git ref (branch, tag, or commit SHA).

        Returns:
            A CodeSearchItem for the first matching file, or None if not found.
        """
        owner, name = self._split_repo(repo)
        # Use path: syntax for filename search
        search_query = f"filename:{filename} repo:{owner}/{name}"
        url = "https://api.github.com/search/code"
        params = {"q": search_query, "per_page": 1}
        if ref:
            params["ref"] = ref
        headers = dict(self.headers)
        headers["Accept"] = "application/vnd.github.text-match+json"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(url, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()
            items = data.get("items", [])
            if not items:
                logging.debug("No items found for: %s", search_query)
                return None
            logging.debug("Found %d items for: %s", len(items), search_query)
            item = items[0]
            repo_nwo = item["repository"]["full_name"]
            path = item["path"]
            is_binary = False
            byte_size = item.get("size")
            text: str | None = None
            text_matches = item.get("text_matches")
            snippet = None
            if text_matches and isinstance(text_matches, list):
                snippet = "\n".join(
                    [m.get("fragment", "") for m in text_matches if m.get("fragment")]
                )
            if include_text:
                contents_url = item["url"].replace("/search/code", "/repos")
                file_resp = await client.get(contents_url, headers=self.headers)
                file_resp.raise_for_status()
                file_data = file_resp.json()
                if file_data.get("encoding") == "base64":
                    import base64

                    try:
                        text = base64.b64decode(file_data["content"]).decode(
                            "utf-8", errors="replace"
                        )
                    except Exception:
                        is_binary = True
                        text = None
                else:
                    text = file_data.get("content")
            return CodeSearchItem(
                repository=repo_nwo,
                path=path,
                text=text if include_text and not is_binary else None,
                is_binary=is_binary,
                byte_size=byte_size,
                snippet=snippet,
            )

    async def list_repository_files(
        self,
        repo: str,
        ref: str | None = None,
    ) -> List[str]:
        """Return a list of all file paths (blobs) in the repository at the given ref.

        Uses the Git Trees REST API:
            1. Resolve default branch if ref not supplied.
            2. GET /repos/{owner}/{repo}/git/trees/{ref}?recursive=1

        Args:
            repo: Repository in the form "owner/name".
            ref: Branch / tag / commit SHA (defaults to default branch if omitted).

        Returns:
            A list of file paths (strings). If the underlying tree response is truncated
            (very large repo), all paths returned by GitHub are included, and a warning is logged.
        """
        owner, name = self._split_repo(repo)
        headers = dict(self.headers)
        api_base = "https://api.github.com"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            if not ref:
                repo_resp = await client.get(f"{api_base}/repos/{owner}/{name}", headers=headers)
                repo_resp.raise_for_status()
                ref = repo_resp.json().get("default_branch") or "main"

            tree_resp = await client.get(
                f"{api_base}/repos/{owner}/{name}/git/trees/{ref}",
                params={"recursive": 1},
                headers=headers,
            )
            tree_resp.raise_for_status()
            tree_json = tree_resp.json()
            truncated = bool(tree_json.get("truncated"))
            paths: List[str] = [e.get("path") for e in tree_json.get("tree", []) if e.get("type") == "blob" and e.get("path")]
            if truncated:
                logging.warning(
                    "Tree listing truncated for %s (ref=%s); results may be incomplete.",
                    repo,
                    ref,
                )
            return paths

    async def list_repository_labels(
        self,
        repo: str,
        page_size: int = 50,
        max_labels: int | None = None,
    ) -> List[Dict[str, Any]]:
        """List all labels for a repository using the REST endpoint.

        Endpoint: GET /repos/{owner}/{repo}/labels (paginated via Link headers).

        Args:
            repo: Repository full name ("owner/name").
            page_size: Page size per request (max 100 per GitHub docs; enforced here).
            max_labels: Optional cap on total labels returned; fetches all when None.

        Returns:
            List of dicts: {name, description, color}.
        """
        owner, name = self._split_repo(repo)
        per_page = min(100, max(1, page_size))
        page = 1
        labels: List[Dict[str, Any]] = []
        base_url = f"https://api.github.com/repos/{owner}/{name}/labels"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            while True:
                params = {"per_page": per_page, "page": page}
                resp = await client.get(base_url, headers=self.headers, params=params)
                resp.raise_for_status()
                batch = resp.json()
                if not isinstance(batch, list):
                    break
                for item in batch:
                    labels.append({
                        "name": item.get("name"),
                        "description": (item.get("description") or "").strip(),
                        "color": item.get("color") or "",
                    })
                    if max_labels is not None and len(labels) >= max_labels:
                        return labels
                # Pagination: check Link header for rel="next"
                link_header = resp.headers.get("Link") or ""
                if 'rel="next"' not in link_header:
                    break
                page += 1
        return labels
