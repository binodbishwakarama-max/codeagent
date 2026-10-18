"""GitHub REST API client for fetching diffs, repository contents, and managing reviews."""

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

GITHUB_API_BASE = "https://api.github.com"


class GitHubClient:
    """Asynchronous HTTP client for interacting with the GitHub REST API."""

    def __init__(
        self,
        token: str,
        client: httpx.AsyncClient | None = None,
        base_url: str = GITHUB_API_BASE,
    ) -> None:
        self.token = token
        self.base_url = base_url.rstrip("/")
        self._external_client = client
        self._client: httpx.AsyncClient | None = client

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def close(self) -> None:
        """Close internal HTTP client if created internally."""
        if self._external_client is None and self._client is not None:
            await self._client.aclose()
            self._client = None

    def _default_headers(self, accept: str = "application/vnd.github+json") -> dict[str, str]:
        return {
            "Accept": accept,
            "Authorization": f"Bearer {self.token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    async def fetch_pull_request_diff(self, repo_full_name: str, pr_number: int) -> str:
        """Fetch unified diff for a pull request."""
        client = await self._get_client()
        url = f"{self.base_url}/repos/{repo_full_name}/pulls/{pr_number}"
        headers = self._default_headers(accept="application/vnd.github.v3.diff")
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        return response.text

    async def fetch_repo_file(
        self, repo_full_name: str, path: str, ref: str | None = None
    ) -> str | None:
        """Fetch raw file content from a repository (e.g. .prism/standards.yml)."""
        client = await self._get_client()
        url = f"{self.base_url}/repos/{repo_full_name}/contents/{path}"
        params: dict[str, str] = {}
        if ref:
            params["ref"] = ref
        headers = self._default_headers(accept="application/vnd.github.raw+json")
        response = await client.get(url, headers=headers, params=params)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.text

    async def list_pull_request_reviews(
        self, repo_full_name: str, pr_number: int
    ) -> list[dict[str, Any]]:
        """List reviews submitted for a pull request."""
        client = await self._get_client()
        url = f"{self.base_url}/repos/{repo_full_name}/pulls/{pr_number}/reviews"
        headers = self._default_headers()
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        return []

    async def list_pull_request_comments(
        self,
        repo_full_name: str,
        pr_number: int,
    ) -> list[dict[str, Any]]:
        """List review comments on a pull request."""
        client = await self._get_client()
        url = f"{self.base_url}/repos/{repo_full_name}/pulls/{pr_number}/comments"
        headers = self._default_headers()
        response = await client.get(url, headers=headers)
        if response.is_error:
            logger.warning(
                "Failed to list PR comments for %s#%d: %s",
                repo_full_name,
                pr_number,
                response.text,
            )
            return []
        data = response.json()
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        return []

    async def dismiss_pull_request_review(
        self,
        repo_full_name: str,
        pr_number: int,
        review_id: int,
        message: str = "Superceded by newer Prism review.",
    ) -> None:
        """Dismiss an earlier bot review."""
        client = await self._get_client()
        url = (
            f"{self.base_url}/repos/{repo_full_name}/pulls"
            f"/{pr_number}/reviews/{review_id}/dismissals"
        )
        headers = self._default_headers()
        payload = {"message": message}
        response = await client.put(url, headers=headers, json=payload)
        # Note: 422 or 404 may occur if already dismissed, log without failing
        if response.is_error and response.status_code != 422:
            logger.warning(
                "Failed to dismiss review %d on %s#%d: %s",
                review_id,
                repo_full_name,
                pr_number,
                response.text,
            )

    async def create_pull_request_review(
        self,
        repo_full_name: str,
        pr_number: int,
        commit_id: str,
        body: str,
        comments: list[dict[str, Any]],
        event: str = "COMMENT",
    ) -> dict[str, Any]:
        """Submit a pull request review with inline comments and summary body."""
        client = await self._get_client()
        url = f"{self.base_url}/repos/{repo_full_name}/pulls/{pr_number}/reviews"
        headers = self._default_headers()
        payload: dict[str, Any] = {
            "commit_id": commit_id,
            "body": body,
            "event": event,
            "comments": comments,
        }
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {}

    async def create_check_run(
        self,
        repo_full_name: str,
        head_sha: str,
        name: str = "Prism Review",
        status: str = "in_progress",
    ) -> int:
        """Create a GitHub check run and return its ID."""
        client = await self._get_client()
        url = f"{self.base_url}/repos/{repo_full_name}/check-runs"
        headers = self._default_headers()
        payload: dict[str, Any] = {
            "name": name,
            "head_sha": head_sha,
            "status": status,
        }
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return int(data.get("id", 0)) if isinstance(data, dict) else 0

    async def update_check_run(
        self,
        repo_full_name: str,
        check_run_id: int,
        status: str = "completed",
        conclusion: str = "success",
        summary: str = "",
    ) -> None:
        """Update an existing check run with a final conclusion.

        Conclusion values: success, neutral, failure.
        """
        client = await self._get_client()
        url = f"{self.base_url}/repos/{repo_full_name}/check-runs/{check_run_id}"
        headers = self._default_headers()
        payload: dict[str, Any] = {
            "status": status,
            "conclusion": conclusion,
            "output": {
                "title": "Prism Review",
                "summary": summary,
            },
        }
        response = await client.patch(url, headers=headers, json=payload)
        if response.is_error:
            logger.warning(
                "Failed to update check run %d on %s: %s",
                check_run_id,
                repo_full_name,
                response.text,
            )

    async def get_pull_request_comment(
        self,
        repo_full_name: str,
        comment_id: int,
    ) -> dict[str, Any] | None:
        """Fetch a specific pull request review comment by ID."""
        client = await self._get_client()
        url = f"{self.base_url}/repos/{repo_full_name}/pulls/comments/{comment_id}"
        headers = self._default_headers()
        response = await client.get(url, headers=headers)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else None

    async def reply_to_pull_request_comment(
        self,
        repo_full_name: str,
        pr_number: int,
        comment_id: int,
        body: str,
    ) -> dict[str, Any]:
        """Post an inline reply to an existing pull request review comment thread."""
        client = await self._get_client()
        url = f"{self.base_url}/repos/{repo_full_name}/pulls/{pr_number}/comments/{comment_id}/replies"
        headers = self._default_headers()
        payload = {"body": body}
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {}

    async def create_issue_comment(
        self,
        repo_full_name: str,
        issue_number: int,
        body: str,
    ) -> dict[str, Any]:
        """Post a top-level comment on an issue or pull request."""
        client = await self._get_client()
        url = f"{self.base_url}/repos/{repo_full_name}/issues/{issue_number}/comments"
        headers = self._default_headers()
        payload = {"body": body}
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {}
