"""Memory backends for pluggable file storage."""

from deepagents.backends.composite import CompositeBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.state import StateBackend
from deepagents.backends.store import StoreBackend
from deepagents.backends.protocol import BackendProtocol

# Sandbox is optional - only import if needed
try:
    from deepagents.backends.sandbox import (
        SandboxBackend,
        SandboxConfig,
        SandboxProvider,
        ModalSandboxProvider,
        DaytonaProvider,
    )
    _SANDBOX_AVAILABLE = True
except ImportError:
    _SANDBOX_AVAILABLE = False

__all__ = [
    "BackendProtocol",
    "CompositeBackend",
    "FilesystemBackend",
    "StateBackend",
    "StoreBackend",
]

if _SANDBOX_AVAILABLE:
    __all__.extend([
        "SandboxBackend",
        "SandboxConfig",
        "SandboxProvider",
        "ModalSandboxProvider",
        "DaytonaProvider",
    ])
