"""Daytona sandbox provider implementation"""

import httpx
from deepagents.backends.sandbox.protocol import (
    SandboxProvider,
    SandboxConfig,
    ExecutionResult,
    FileMetadata,
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
