"""Type definitions for sandbox backends"""

from typing import Literal, TypedDict
from dataclasses import dataclass


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
