"""Tests for SandboxBackend and providers"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from deepagents.backends.sandbox import (
    SandboxBackend,
    SandboxConfig,
    DaytonaProvider,
)
from deepagents.backends.sandbox.protocol import ExecutionResult, FileMetadata


class MockSandboxProvider:
    """Mock provider for testing SandboxBackend"""

    def __init__(self):
        self.files = {}

    async def read_file(self, path: str) -> bytes:
        if path in self.files:
            return self.files[path]
        raise FileNotFoundError(f"File not found: {path}")

    async def write_file(self, path: str, content: bytes) -> None:
        self.files[path] = content

    async def list_files(self, path: str, recursive: bool = False) -> list[FileMetadata]:
        files = []
        for file_path in self.files.keys():
            if file_path.startswith(path):
                files.append(
                    {
                        "path": file_path,
                        "size": len(self.files[file_path]),
                        "is_dir": False,
                        "modified_at": "2025-01-01T00:00:00Z",
                    }
                )
        return files

    async def delete_file(self, path: str) -> None:
        if path in self.files:
            del self.files[path]

    async def file_exists(self, path: str) -> bool:
        return path in self.files

    async def create_directory(self, path: str) -> None:
        pass

    async def get_file_metadata(self, path: str) -> FileMetadata:
        if path in self.files:
            return {
                "path": path,
                "size": len(self.files[path]),
                "is_dir": False,
                "modified_at": "2025-01-01T00:00:00Z",
            }
        raise FileNotFoundError(f"File not found: {path}")

    async def execute(
        self, command: str, cwd: str = "/workspace", env: dict[str, str] | None = None
    ) -> ExecutionResult:
        # Simple mock execution
        return {"stdout": "mock output", "stderr": "", "exit_code": 0}


class TestSandboxBackend:
    """Test SandboxBackend implementation"""

    def test_read_file_success(self):
        """Test reading a file from sandbox"""
        provider = MockSandboxProvider()
        provider.files["/workspace/test.txt"] = b"line1\nline2\nline3"

        backend = SandboxBackend(provider)
        result = backend.read("/workspace/test.txt")

        assert "line1" in result
        assert "line2" in result
        assert "line3" in result

    def test_read_file_not_found(self):
        """Test reading a non-existent file"""
        provider = MockSandboxProvider()
        backend = SandboxBackend(provider)

        result = backend.read("/workspace/missing.txt")
        assert "Error" in result
        assert "Could not read file" in result

    def test_write_file_success(self):
        """Test writing a file to sandbox"""
        provider = MockSandboxProvider()
        backend = SandboxBackend(provider)

        result = backend.write("/workspace/new.txt", "Hello World")

        assert result.error is None
        assert result.path == "/workspace/new.txt"
        assert result.files_update is None  # Sandbox doesn't use state updates
        assert provider.files["/workspace/new.txt"] == b"Hello World"

    def test_edit_file_success(self):
        """Test editing a file in sandbox"""
        provider = MockSandboxProvider()
        provider.files["/workspace/test.txt"] = b"Hello World"

        backend = SandboxBackend(provider)
        result = backend.edit("/workspace/test.txt", "World", "Universe", False)

        assert result.error is None
        assert result.path == "/workspace/test.txt"
        assert result.occurrences == 1
        assert provider.files["/workspace/test.txt"] == b"Hello Universe"

    def test_edit_file_not_found(self):
        """Test editing a non-existent file"""
        provider = MockSandboxProvider()
        backend = SandboxBackend(provider)

        result = backend.edit("/workspace/missing.txt", "old", "new", False)
        assert result.error is not None
        assert "Could not read file" in result.error

    def test_ls_info(self):
        """Test listing files"""
        provider = MockSandboxProvider()
        provider.files["/workspace/file1.txt"] = b"content1"
        provider.files["/workspace/file2.txt"] = b"content2"

        backend = SandboxBackend(provider)
        files = backend.ls_info("/workspace/")

        assert len(files) == 2
        paths = [f["path"] for f in files]
        assert "/workspace/file1.txt" in paths
        assert "/workspace/file2.txt" in paths


class TestSandboxConfig:
    """Test SandboxConfig"""

    def test_modal_config(self):
        """Test creating Modal configuration"""
        config = SandboxConfig(
            provider="modal",
            image="python:3.11-slim",
            timeout_seconds=1800,
            memory_mb=4096,
            cpu_count=2.0,
        )

        assert config.provider == "modal"
        assert config.image == "python:3.11-slim"
        assert config.timeout_seconds == 1800
        assert config.memory_mb == 4096
        assert config.cpu_count == 2.0

    def test_daytona_config(self):
        """Test creating Daytona configuration"""
        config = SandboxConfig(
            provider="daytona",
            workspace_id="ws-123",
            api_key="test-key",
            api_url="https://api.test.com",
        )

        assert config.provider == "daytona"
        assert config.workspace_id == "ws-123"
        assert config.api_key == "test-key"
        assert config.api_url == "https://api.test.com"


class TestDaytonaProvider:
    """Test DaytonaProvider"""

    def test_daytona_config_validation(self):
        """Test Daytona configuration validation"""
        config = SandboxConfig(provider="daytona")

        with pytest.raises(ValueError, match="workspace_id required"):
            DaytonaProvider.from_config(config)

        config.workspace_id = "ws-123"
        with pytest.raises(ValueError, match="api_key required"):
            DaytonaProvider.from_config(config)

    @pytest.mark.asyncio
    async def test_daytona_execute(self):
        """Test executing command via Daytona"""
        config = SandboxConfig(
            provider="daytona",
            workspace_id="ws-123",
            api_key="test-key",
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.json.return_value = {
                "stdout": "test output",
                "stderr": "",
                "exit_code": 0,
            }
            mock_client.return_value.post = AsyncMock(return_value=mock_response)

            provider = DaytonaProvider(config)
            result = await provider.execute("echo hello")

            assert result["stdout"] == "test output"
            assert result["exit_code"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
