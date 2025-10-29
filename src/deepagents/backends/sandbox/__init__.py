"""Sandbox backend for remote code execution"""

from deepagents.backends.protocol import BackendProtocol

# Import types FIRST (before providers to avoid circular import)
from deepagents.backends.sandbox.types import (
    ExecutionResult,
    FileMetadata,
    BootstrapConfig,
    SandboxConfig,
)

# Now import providers (they can import from types.py without circular dependency)
from deepagents.backends.sandbox.providers import (
    ModalSandboxProvider,
    DaytonaProvider,
)


def create_sandbox_provider(config: SandboxConfig) -> BackendProtocol:
    """Factory: creates the appropriate sandbox provider based on config.

    Args:
        config: SandboxConfig with provider field set to "modal" or "daytona".

    Returns:
        BackendProtocol instance (ModalSandboxProvider or DaytonaProvider)

    Raises:
        ValueError: If provider is unknown

    Example:
        config = SandboxConfig(provider="modal")
        provider = create_sandbox_provider(config)
        # provider now implements BackendProtocol + has execute() method
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
    "SandboxConfig",
    "BootstrapConfig",
    "ExecutionResult",
    "FileMetadata",
    "create_sandbox_provider",
    "ModalSandboxProvider",
    "DaytonaProvider",
]
