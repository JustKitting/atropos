"""
JJ Executor — HTTP Client for the JJ Execution Server
======================================================

Async HTTP client that talks to the Dockerized jj server.
All jj binary execution happens inside the container — this side
just sends HTTP requests and parses responses.

The server must be running:
    docker build -t jj-exec-server .
    docker run -p 5003:5003 jj-exec-server
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_SERVER_URL = "http://localhost:5003"


@dataclass
class JJResult:
    """Result of a jj command execution."""

    command: str
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool = False


@dataclass
class RepoState:
    """Captured state of a jj repository for scoring."""

    file_contents: Dict[str, str] = field(default_factory=dict)
    log_output: str = ""
    status_output: str = ""
    diff_output: str = ""
    has_conflicts: bool = False
    is_empty: bool = True


class JJExecutor:
    """Async HTTP client for the jj execution server."""

    def __init__(
        self,
        server_url: str = DEFAULT_SERVER_URL,
        command_timeout: int = 30,
    ):
        import httpx  # noqa: F811 — deferred import so dataclasses are importable without httpx

        self._httpx = httpx
        self.server_url = server_url.rstrip("/")
        self.command_timeout = command_timeout

    async def health_check(self) -> bool:
        """Check if the jj server is running and jj is available."""
        try:
            async with self._httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.server_url}/health")
                data = resp.json()
                if data.get("status") == "ok":
                    logger.info(f"JJ server healthy: {data.get('jj_version')}")
                    return True
                logger.error(f"JJ server unhealthy: {data}")
                return False
        except self._httpx.ConnectError:
            logger.error(
                f"Cannot connect to jj server at {self.server_url}. "
                "Is the Docker container running?\n"
                "  docker build -t jj-exec-server environments/community/jj_env/\n"
                "  docker run -p 5003:5003 jj-exec-server"
            )
            return False
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def create_repo(self) -> str:
        """
        Create a new jj repo on the server. Returns repo_id.
        Raises RuntimeError if the server is unavailable.
        """
        try:
            async with self._httpx.AsyncClient(timeout=self.command_timeout) as client:
                resp = await client.post(f"{self.server_url}/repo/create")
                data = resp.json()
                if "error" in data:
                    raise RuntimeError(f"Failed to create repo: {data['error']}")
                return data["repo_id"]
        except self._httpx.ConnectError:
            raise RuntimeError(
                f"Cannot connect to jj server at {self.server_url}. "
                "Please ensure the Docker container is running:\n"
                "  docker build -t jj-exec-server environments/community/jj_env/\n"
                "  docker run -p 5003:5003 jj-exec-server"
            )

    async def setup_scrambled_repo(
        self,
        repo_id: str,
        scramble_ops: List[Dict],
    ) -> RepoState:
        """
        Execute scramble operations on a repo to create a broken state.
        Returns the resulting RepoState.
        """
        async with self._httpx.AsyncClient(timeout=self.command_timeout * 2) as client:
            resp = await client.post(
                f"{self.server_url}/repo/{repo_id}/setup",
                json={"scramble_ops": scramble_ops},
            )
            data = resp.json()
            if "error" in data:
                raise RuntimeError(f"Setup failed: {data['error']}")
            return self._parse_state(data["state"])

    async def execute_model_command(
        self,
        repo_id: str,
        command_str: str,
    ) -> JJResult:
        """
        Execute a model-issued jj command (without "jj" prefix).
        Validation happens server-side.
        """
        try:
            async with self._httpx.AsyncClient(timeout=self.command_timeout) as client:
                resp = await client.post(
                    f"{self.server_url}/repo/{repo_id}/execute",
                    json={"command": command_str},
                )
                data = resp.json()
                return JJResult(
                    command=f"jj {command_str}",
                    stdout=data.get("stdout", ""),
                    stderr=data.get("stderr", ""),
                    returncode=data.get("returncode", -1),
                    timed_out=data.get("timed_out", False),
                )
        except self._httpx.TimeoutException:
            return JJResult(
                command=f"jj {command_str}",
                stdout="",
                stderr=f"HTTP timeout after {self.command_timeout}s",
                returncode=-1,
                timed_out=True,
            )
        except Exception as e:
            return JJResult(
                command=f"jj {command_str}",
                stdout="",
                stderr=f"Request error: {e}",
                returncode=-1,
                timed_out=False,
            )

    async def write_file_in_repo(
        self,
        repo_id: str,
        path: str,
        content: str,
    ) -> JJResult:
        """Write a file in the repo (for conflict resolution)."""
        try:
            async with self._httpx.AsyncClient(timeout=self.command_timeout) as client:
                resp = await client.post(
                    f"{self.server_url}/repo/{repo_id}/write_file",
                    json={"path": path, "content": content},
                )
                data = resp.json()
                return JJResult(
                    command=f"write_file {path}",
                    stdout=data.get("stdout", ""),
                    stderr=data.get("stderr", ""),
                    returncode=data.get("returncode", -1),
                )
        except Exception as e:
            return JJResult(
                command=f"write_file {path}",
                stdout="",
                stderr=f"Request error: {e}",
                returncode=-1,
            )

    async def capture_state(self, repo_id: str) -> RepoState:
        """Capture the current repo state for scoring."""
        async with self._httpx.AsyncClient(timeout=self.command_timeout) as client:
            resp = await client.get(f"{self.server_url}/repo/{repo_id}/state")
            data = resp.json()
            return self._parse_state(data)

    async def cleanup_repo(self, repo_id: str) -> None:
        """Delete a repo on the server."""
        try:
            async with self._httpx.AsyncClient(timeout=10.0) as client:
                await client.delete(f"{self.server_url}/repo/{repo_id}/cleanup")
        except Exception as e:
            logger.warning(f"Failed to cleanup repo {repo_id}: {e}")

    async def get_initial_prompt_info(self, repo_id: str) -> Dict[str, str]:
        """
        Gather repo info to include in the user prompt.
        """
        state = await self.capture_state(repo_id)

        file_listing = []
        for path, content in sorted(state.file_contents.items()):
            file_listing.append(f"=== {path} ===\n{content}")

        return {
            "log": state.log_output,
            "status": state.status_output,
            "diff": state.diff_output,
            "files": "\n\n".join(file_listing) if file_listing else "(no files)",
        }

    @staticmethod
    def _parse_state(data: dict) -> RepoState:
        """Parse a state dict from the server into a RepoState."""
        return RepoState(
            file_contents=data.get("file_contents", {}),
            log_output=data.get("log_output", ""),
            status_output=data.get("status_output", ""),
            diff_output=data.get("diff_output", ""),
            has_conflicts=data.get("has_conflicts", False),
            is_empty=data.get("is_empty", True),
        )
