"""Protocol and types for sandbox providers"""

from typing import Protocol, Literal, TypedDict
from dataclasses import dataclass, field


class ExecutionResult(TypedDict):
    """Result of command execution in sandbox"""
    stdout: str
    stderr: str
    exit_code: int


class FileMetadata(TypedDict):
    """Metadata about a file in sandbox"""
    path: str
    size: int
    is_dir: bool
    modified_at: str


@dataclass
class BootstrapConfig:
    """Configuration for bootstrapping sandbox environment"""

    # Git repository to clone
    git_repo: str | None = None
    git_branch: str | None = None

    # Local files to copy into sandbox
    local_files: dict[str, str] | None = None  # {local_path: sandbox_path}

    # Setup script to run after bootstrap
    setup_script: str | None = None

    # Working directory for operations
    workdir: str = "/workspace"


@dataclass
class SandboxConfig:
    """Configuration for sandbox providers"""

    # Provider selection
    provider: Literal["modal", "daytona"]

    # Modal Sandbox specific
    image: str = "python:3.11-slim"
    timeout_seconds: int = 3600  # How long sandbox can stay alive

    # Resource limits
    memory_mb: int = 2048
    cpu_count: float = 1.0

    # Environment
    env_vars: dict[str, str] | None = None

    # Bootstrap configuration
    bootstrap: BootstrapConfig | None = None

    # Daytona specific
    workspace_id: str | None = None
    api_key: str | None = None
    api_url: str | None = None

    # Working directory
    workdir: str = "/workspace"


class SandboxProvider(Protocol):
    """Protocol that all sandbox providers must implement"""

    @property
    def sandbox_id(self) -> str | None:
        """Get the sandbox ID for tracking and cleanup"""
        ...

    async def ensure_ready(self) -> None:
        """Ensure sandbox is created and ready (lazy initialization)"""
        ...

    async def execute(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """Execute a shell command in the sandbox"""
        ...

    async def read_file(self, path: str) -> bytes:
        """Read file contents from sandbox"""
        ...

    async def write_file(self, path: str, content: bytes) -> None:
        """Write file contents to sandbox"""
        ...

    async def list_files(
        self,
        path: str,
        recursive: bool = False,
    ) -> list[FileMetadata]:
        """List files in directory"""
        ...

    async def delete_file(self, path: str) -> None:
        """Delete a file"""
        ...

    async def file_exists(self, path: str) -> bool:
        """Check if file exists"""
        ...

    async def create_directory(self, path: str) -> None:
        """Create directory"""
        ...

    async def get_file_metadata(self, path: str) -> FileMetadata:
        """Get file metadata"""
        ...

    async def cleanup(self) -> None:
        """Clean up sandbox resources (terminate container, delete data)"""
        ...
