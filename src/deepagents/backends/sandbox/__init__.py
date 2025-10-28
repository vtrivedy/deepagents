"""Sandbox backend for remote code execution"""

from deepagents.backends.sandbox.protocol import (
    SandboxProvider,
    SandboxConfig,
    BootstrapConfig,
    ExecutionResult,
    FileMetadata,
)
from deepagents.backends.sandbox.backend import SandboxBackend
from deepagents.backends.sandbox.providers import (
    ModalSandboxProvider,
    DaytonaProvider,
)


def create_sandbox_provider(config: SandboxConfig) -> SandboxProvider:
    """Factory: creates the appropriate sandbox provider based on config.

    Args:
        config: SandboxConfig with provider field set to "modal" or "daytona".

    Returns:
        SandboxProvider instance of the appropriate type

    Raises:
        ValueError: If provider is unknown

    Example:
        config = SandboxConfig(provider="modal")
        provider = create_sandbox_provider(config)
        backend = SandboxBackend(provider)
    """
    if config.provider == "modal":
        return ModalSandboxProvider.from_config(config)

    elif config.provider == "daytona":
        return DaytonaProvider.from_config(config)

    else:
        raise ValueError(
            f"Unknown sandbox provider: {config.provider}. "
            f"Supported providers: modal, daytona"
        )


__all__ = [
    "SandboxProvider",
    "SandboxConfig",
    "BootstrapConfig",
    "ExecutionResult",
    "FileMetadata",
    "SandboxBackend",
    "create_sandbox_provider",
    "ModalSandboxProvider",
    "DaytonaProvider",
]
