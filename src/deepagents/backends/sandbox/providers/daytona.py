"""Daytona sandbox provider implementation"""

import httpx
import asyncio
import json
from typing import Optional
from deepagents.backends.sandbox.types import (
    SandboxConfig,
    ExecutionResult,
    FileMetadata,
)
from deepagents.backends.protocol import WriteResult, EditResult
from deepagents.backends.utils import (
    FileInfo,
    GrepMatch,
    format_content_with_line_numbers,
    perform_string_replacement,
)


class DaytonaProvider:
    """Daytona sandbox provider"""

    def __init__(self, config: SandboxConfig):
        self.config = config
        self.workspace_id = config.workspace_id
        self.base_url = config.api_url or "https://api.daytona.io/v1"

        self.client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
            timeout=config.timeout_seconds,
        )

    @classmethod
    def from_config(cls, config: SandboxConfig) -> "DaytonaProvider":
        """Create from configuration"""
        if not config.workspace_id:
            raise ValueError("workspace_id required for Daytona")
        if not config.api_key:
            raise ValueError("api_key required for Daytona")
        return cls(config)

    async def execute(
        self,
        command: str,
        cwd: str = "/workspace",
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """Execute command in Daytona workspace"""
        response = await self.client.post(
            f"{self.base_url}/workspaces/{self.workspace_id}/exec",
            json={
                "command": command,
                "cwd": cwd,
                "env": env or {},
            },
        )
        response.raise_for_status()
        data = response.json()

        return {
            "stdout": data.get("stdout", ""),
            "stderr": data.get("stderr", ""),
            "exit_code": data.get("exit_code", 0),
        }

    async def read_file(self, path: str) -> bytes:
        """Read file from Daytona workspace"""
        response = await self.client.get(
            f"{self.base_url}/workspaces/{self.workspace_id}/files",
            params={"path": path},
        )
        response.raise_for_status()
        return response.content

    async def write_file(self, path: str, content: bytes) -> None:
        """Write file to Daytona workspace"""
        response = await self.client.post(
            f"{self.base_url}/workspaces/{self.workspace_id}/files",
            params={"path": path},
            content=content,
        )
        response.raise_for_status()

    async def list_files(
        self,
        path: str = "/workspace",
        recursive: bool = False,
    ) -> list[FileMetadata]:
        """List files in Daytona workspace"""
        response = await self.client.get(
            f"{self.base_url}/workspaces/{self.workspace_id}/files",
            params={"path": path, "recursive": recursive},
        )
        response.raise_for_status()
        data = response.json()

        return [
            {
                "path": f["path"],
                "size": f["size"],
                "is_dir": f["is_directory"],
                "modified_at": f["modified_at"],
            }
            for f in data.get("files", [])
        ]

    async def delete_file(self, path: str) -> None:
        """Delete file from Daytona workspace"""
        response = await self.client.delete(
            f"{self.base_url}/workspaces/{self.workspace_id}/files",
            params={"path": path},
        )
        response.raise_for_status()

    async def file_exists(self, path: str) -> bool:
        """Check if file exists in Daytona workspace"""
        try:
            response = await self.client.head(
                f"{self.base_url}/workspaces/{self.workspace_id}/files",
                params={"path": path},
            )
            return response.status_code == 200
        except httpx.HTTPStatusError:
            return False

    async def create_directory(self, path: str) -> None:
        """Create directory in Daytona workspace"""
        response = await self.client.post(
            f"{self.base_url}/workspaces/{self.workspace_id}/directories",
            json={"path": path},
        )
        response.raise_for_status()

    async def get_file_metadata(self, path: str) -> FileMetadata:
        """Get file metadata from Daytona workspace"""
        response = await self.client.get(
            f"{self.base_url}/workspaces/{self.workspace_id}/files/metadata",
            params={"path": path},
        )
        response.raise_for_status()
        data = response.json()

        return {
            "path": data["path"],
            "size": data["size"],
            "is_dir": data["is_directory"],
            "modified_at": data["modified_at"],
        }

    async def cleanup(self) -> None:
        """Clean up Daytona client"""
        await self.client.aclose()

    def __enter__(self) -> "DaytonaProvider":
        """Context manager entry - enables `with provider:` syntax"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically cleanup client"""
        import asyncio
        asyncio.run(self.cleanup())
        return False

    # ============================================================================
    # BackendProtocol Implementation
    # ============================================================================
    # These methods implement the BackendProtocol interface, allowing Daytona
    # workspace to be used as a backend for filesystem operations.

    def _get_loop(self):
        """Get or create event loop for sync→async bridge"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop

    def _run_async(self, coro):
        """Run async coroutine in sync context (for BackendProtocol)"""
        loop = self._get_loop()
        return loop.run_until_complete(coro)

    def ls_info(self, path: str) -> list[FileInfo]:
        """List directory contents (BackendProtocol method)"""
        files = self._run_async(self.list_files(path, recursive=False))

        return [
            FileInfo(
                path=f["path"],
                size=f["size"],
                is_dir=f["is_dir"],
                modified_at=f["modified_at"],
            )
            for f in files
        ]

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """Read file with line numbers (BackendProtocol method)"""
        try:
            content_bytes = self._run_async(self.read_file(file_path))
            content = content_bytes.decode("utf-8")
        except Exception as e:
            return f"Error: Could not read file '{file_path}': {str(e)}"

        lines = content.split("\n")

        # Apply offset and limit
        if offset > 0:
            lines = lines[offset:]
        if limit > 0 and len(lines) > limit:
            lines = lines[:limit]

        # Format with line numbers
        return format_content_with_line_numbers(lines, offset + 1)

    def write(self, file_path: str, content: str) -> WriteResult:
        """Write file to workspace (BackendProtocol method)"""
        try:
            self._run_async(
                self.write_file(file_path, content.encode("utf-8"))
            )

            return WriteResult(
                path=file_path,
                files_update=None,
            )
        except Exception as e:
            return WriteResult(
                error=f"Failed to write file '{file_path}': {str(e)}",
            )

    def upload_file(
        self,
        path: str,
        content: Optional[bytes | str] = None,
        local_path: Optional[str] = None,
    ) -> WriteResult:
        """Upload file to workspace (convenience method)"""
        if content is None and local_path is None:
            raise ValueError("Must specify either 'content' or 'local_path'")

        if content is not None and local_path is not None:
            raise ValueError("Specify either 'content' or 'local_path', not both")

        # Read from local file if specified
        if local_path is not None:
            try:
                with open(local_path, 'rb') as f:
                    content = f.read()
            except Exception as e:
                return WriteResult(
                    error=f"Failed to read local file '{local_path}': {str(e)}",
                )

        # Convert string to bytes if needed
        if isinstance(content, str):
            content = content.encode('utf-8')

        # Upload to workspace
        try:
            self._run_async(self.write_file(path, content))

            return WriteResult(
                path=path,
                files_update=None,
            )
        except Exception as e:
            return WriteResult(
                error=f"Failed to upload file '{path}': {str(e)}",
            )

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Edit file in workspace using string replacement (BackendProtocol method)"""
        try:
            # Read current content
            content_bytes = self._run_async(self.read_file(file_path))
            content = content_bytes.decode("utf-8")
        except Exception as e:
            return EditResult(
                error=f"Error: Could not read file '{file_path}': {str(e)}",
            )

        # Perform replacement
        result = perform_string_replacement(
            content, old_string, new_string, replace_all
        )

        if isinstance(result, str):
            # Error message
            return EditResult(error=result)

        new_content, num_replacements = result

        # Write back
        try:
            self._run_async(
                self.write_file(file_path, new_content.encode("utf-8"))
            )

            return EditResult(
                path=file_path,
                files_update=None,
                occurrences=int(num_replacements),
            )
        except Exception as e:
            return EditResult(
                error=f"Failed to write file '{file_path}': {str(e)}",
            )

    def grep_raw(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob: Optional[str] = None,
    ) -> list[GrepMatch] | str:
        """Search for pattern in files (BackendProtocol method)

        Note: This requires grep/ripgrep installed in the workspace
        """
        search_path = path or "/workspace"

        # Build grep command
        if glob:
            cmd = f"rg --json '{pattern}' --glob '{glob}' {search_path}"
        else:
            cmd = f"rg --json '{pattern}' {search_path}"

        try:
            result = self._run_async(self.execute(cmd, cwd=search_path))
        except Exception as e:
            return f"grep failed: {str(e)}"

        if result["exit_code"] != 0:
            return f"grep failed: {result['stderr']}"

        # Parse ripgrep JSON output
        matches = []
        for line in result["stdout"].split("\n"):
            if not line:
                continue
            try:
                data = json.loads(line)
                if data["type"] == "match":
                    matches.append(
                        GrepMatch(
                            path=data["data"]["path"]["text"],
                            line=data["data"]["line_number"],
                            text=data["data"]["lines"]["text"],
                        )
                    )
            except (json.JSONDecodeError, KeyError):
                continue

        return matches

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        """Find files matching glob pattern (BackendProtocol method)"""
        # Use find command to match glob pattern
        search_path = path.rstrip("/")
        cmd = f"find {search_path} -name '{pattern}'"

        try:
            result = self._run_async(self.execute(cmd, cwd=search_path))
        except Exception:
            return []

        if result["exit_code"] != 0:
            return []

        files = []
        for file_path in result["stdout"].split("\n"):
            if not file_path:
                continue

            try:
                metadata = self._run_async(self.get_file_metadata(file_path))
                files.append(
                    FileInfo(
                        path=metadata["path"],
                        size=metadata["size"],
                        is_dir=metadata["is_dir"],
                        modified_at=metadata["modified_at"],
                    )
                )
            except Exception:
                continue

        return files
