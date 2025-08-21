"""GitHub GraphQL client for repo maintenance tasks.

Functions provided:
- find_stale_open_issues
- close_issue_with_comment
- search_issues_with_bodies
- search_codebase

Authentication:
- Uses a GitHub token with repo read/write and search scopes.
- Reads from env var GITHUB_TOKEN by default; can be passed explicitly.

Notes:
- This client uses GraphQL v4 exclusively.
- All methods are async and use httpx.
"""
from __future__ import annotations

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
            data = await self._graphql(query, {"q": q, "first": page_size, "after": after})
            search = data["search"]
            for node in search["nodes"]:
                if node.get("__typename") != "Issue":
                    continue
                labels = [n["name"] for n in (node.get("labels", {}).get("nodes", []) or [])]
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

    async def close_issue_with_comment(self, repo: str, issue_number: int, comment: str) -> Dict[str, Any]:
        """Post a comment on an issue then close it."""
        owner, name = self._split_repo(repo)
        get_issue_q = """
        query($owner: String!, $name: String!, $number: Int!) {
          repository(owner: $owner, name: $name) {
            issue(number: $number) { id url number title state }
          }
        }
        """
        data = await self._graphql(get_issue_q, {"owner": owner, "name": name, "number": issue_number})
        issue = data["repository"]["issue"]
        if not issue:
            raise RuntimeError(f"Issue #{issue_number} not found in {repo}")
        issue_id = issue["id"]

        add_comment_mut = """
        mutation($input: AddCommentInput!) {
          addComment(input: $input) { clientMutationId }
        }
        """
        await self._graphql(add_comment_mut, {"input": {"subjectId": issue_id, "body": comment}})

        close_issue_mut = """
        mutation($input: CloseIssueInput!) {
          closeIssue(input: $input) { issue { number state } }
        }
        """
        data2 = await self._graphql(close_issue_mut, {"input": {"issueId": issue_id}})
        closed = data2["closeIssue"]["issue"]
        return {"number": closed["number"], "state": closed["state"], "url": issue["url"]}

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
            n = min(page_size, max_comments - len(comments)) if max_comments is not None else page_size
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
            data = await self._graphql(search_q, {"q": q, "first": page_size, "after": after})
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
                        "labels": [n["name"] for n in (it.get("labels", {}).get("nodes", []) or [])],
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
            detail = await self._graphql(comments_q, {"id": it["id"], "n": comments_limit})
            node = detail["node"] or {}
            comments = ((node.get("comments") or {}).get("nodes") or [])
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
                    "labels": [n["name"] for n in (it.get("labels", {}).get("nodes", []) or [])],
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
                    snippet = "\n".join([m.get("fragment", "") for m in text_matches if m.get("fragment")])
                if include_text:
                    # Fetch file content
                    contents_url = item["url"].replace("/search/code", "/repos")
                    file_resp = await client.get(contents_url, headers=self.headers)
                    file_resp.raise_for_status()
                    file_data = file_resp.json()
                    if file_data.get("encoding") == "base64":
                        import base64
                        try:
                            text = base64.b64decode(file_data["content"]).decode("utf-8", errors="replace")
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
                print(f"No items found for: {search_query}")
                return None
            print(f"Found {len(items)} items for: {search_query}")
            item = items[0]
            repo_nwo = item["repository"]["full_name"]
            path = item["path"]
            is_binary = False
            byte_size = item.get("size")
            text: str | None = None
            text_matches = item.get("text_matches")
            snippet = None
            if text_matches and isinstance(text_matches, list):
                snippet = "\n".join([m.get("fragment", "") for m in text_matches if m.get("fragment")])
            if include_text:
                contents_url = item["url"].replace("/search/code", "/repos")
                file_resp = await client.get(contents_url, headers=self.headers)
                file_resp.raise_for_status()
                file_data = file_resp.json()
                if file_data.get("encoding") == "base64":
                    import base64
                    try:
                        text = base64.b64decode(file_data["content"]).decode("utf-8", errors="replace")
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