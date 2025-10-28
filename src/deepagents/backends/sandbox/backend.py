"""SandboxBackend - bridges BackendProtocol to SandboxProvider"""

import asyncio
from typing import Optional
from datetime import datetime

from deepagents.backends.protocol import (
    WriteResult,
    EditResult,
)
from deepagents.backends.utils import (
    FileInfo,
    GrepMatch,
    format_content_with_line_numbers,
    perform_string_replacement,
)
from deepagents.backends.sandbox.protocol import SandboxProvider


class SandboxFilesystem:
    """Filesystem interface for sandbox operations

    Provides a convenient API for file operations in the sandbox.

    Example:
        backend = SandboxBackend(provider)
        fs = backend.fs

        # Upload file
        fs.upload_file(file=b"content", path="/workspace/file.txt")

        # Read file
        content = fs.read("/workspace/file.txt")

        # Edit file
        fs.edit("/workspace/file.txt", old_string="old", new_string="new")

        # List directory
        files = fs.list("/workspace/")
    """

    def __init__(self, backend: 'SandboxBackend'):
        """Initialize filesystem interface

        Args:
            backend: SandboxBackend instance to wrap
        """
        self._backend = backend

    def upload_file(self, file: bytes | str, path: str) -> WriteResult:
        """Upload file content to sandbox

        Args:
            file: File content as bytes or string
            path: Remote path in sandbox

        Returns:
            WriteResult indicating success or error

        Example:
            fs.upload_file(file=b"binary data", path="/workspace/file.bin")
            fs.upload_file(file="text content", path="/workspace/file.txt")
        """
        return self._backend.upload_file(path=path, content=file)

    def read(self, path: str) -> str:
        """Read file from sandbox

        Args:
            path: Path to file in sandbox

        Returns:
            File content formatted with line numbers

        Example:
            content = fs.read("/workspace/script.py")
        """
        return self._backend.read(path)

    def write(self, path: str, content: str) -> WriteResult:
        """Write file to sandbox

        Args:
            path: Path to file in sandbox
            content: File content as string

        Returns:
            WriteResult indicating success or error

        Example:
            fs.write("/workspace/config.json", '{"key": "value"}')
        """
        return self._backend.write(path, content)

    def edit(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Edit file in sandbox using string replacement

        Args:
            path: Path to file in sandbox
            old_string: String to find
            new_string: String to replace with
            replace_all: If True, replace all occurrences. If False, only replace first.

        Returns:
            EditResult with number of replacements made

        Example:
            result = fs.edit(
                "/workspace/code.py",
                old_string="def hello",
                new_string="def greet",
                replace_all=True
            )
            print(f"Made {result.occurrences} replacements")
        """
        return self._backend.edit(path, old_string, new_string, replace_all)

    def list(self, path: str) -> list[FileInfo]:
        """List directory contents

        Args:
            path: Directory path in sandbox

        Returns:
            List of FileInfo objects

        Example:
            files = fs.list("/workspace/")
            for f in files:
                print(f"{f.path} ({f.size} bytes)")
        """
        return self._backend.ls_info(path)


class SandboxBackend:
    """Backend that executes file operations in a remote sandbox

    This implements BackendProtocol by delegating to a SandboxProvider.

    Important: Call cleanup() when done to terminate sandbox and free resources.

    Example:
        provider = ModalSandboxProvider.from_config(config)
        backend = SandboxBackend(provider)

        # Use backend...
        backend.write("/workspace/file.txt", "content")

        # Cleanup when done
        backend.cleanup()

    Or use context manager:
        with SandboxBackend(provider) as backend:
            backend.write("/workspace/file.txt", "content")
        # Cleanup happens automatically
    """

    def __init__(self, provider: SandboxProvider):
        self.provider = provider
        self._loop = None

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup sandbox"""
        self.cleanup()
        return False

    @property
    def sandbox_id(self) -> str | None:
        """Get sandbox ID for tracking"""
        return self.provider.sandbox_id

    @property
    def fs(self) -> SandboxFilesystem:
        """Filesystem interface for convenient file operations

        Returns a filesystem interface object that provides convenient
        methods for file operations.

        Returns:
            SandboxFilesystem instance

        Example:
            backend = SandboxBackend(provider)
            fs = backend.fs

            # Upload file
            fs.upload_file(file=b"content", path="/workspace/file.txt")

            # Read file
            content = fs.read("/workspace/file.txt")

            # Edit file
            fs.edit("/workspace/file.txt", old_string="old", new_string="new")
        """
        if not hasattr(self, '_fs'):
            self._fs = SandboxFilesystem(self)
        return self._fs

    def cleanup(self) -> None:
        """Clean up sandbox resources

        IMPORTANT: Always call this when done with the sandbox!
        This terminates the container and frees resources.
        """
        if hasattr(self.provider, "cleanup"):
            self._run_async(self.provider.cleanup())

    def _get_loop(self):
        """Get or create event loop for sync→async bridge"""
        if self._loop is None:
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        return self._loop

    def _run_async(self, coro):
        """Run async coroutine in sync context (for BackendProtocol)"""
        loop = self._get_loop()
        return loop.run_until_complete(coro)

    def ls_info(self, path: str) -> list[FileInfo]:
        """List directory contents"""
        files = self._run_async(self.provider.list_files(path, recursive=False))

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
        """Read file with line numbers"""
        try:
            content_bytes = self._run_async(self.provider.read_file(file_path))
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
        """Write file to sandbox"""
        try:
            self._run_async(
                self.provider.write_file(file_path, content.encode("utf-8"))
            )

            # Return None because sandbox persists files itself
            # (no need to update LangGraph state)
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
        """Upload file to sandbox

        Upload a file from local filesystem or from bytes/string content.

        Args:
            path: Remote path in sandbox where file will be written
            content: File content as bytes or string. Mutually exclusive with local_path.
            local_path: Path to local file to upload. Mutually exclusive with content.

        Returns:
            WriteResult indicating success or error

        Raises:
            ValueError: If neither or both content and local_path are provided

        Examples:
            # Upload from local file
            backend.upload_file(
                path="/workspace/data.csv",
                local_path="./local_data.csv"
            )

            # Upload from bytes
            backend.upload_file(
                path="/workspace/file.bin",
                content=b"binary data"
            )

            # Upload from string
            backend.upload_file(
                path="/workspace/config.json",
                content='{"key": "value"}'
            )
        """
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

        # Upload to sandbox
        try:
            self._run_async(self.provider.write_file(path, content))

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
        """Edit file in sandbox using string replacement"""
        try:
            # Read current content
            content_bytes = self._run_async(self.provider.read_file(file_path))
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
                self.provider.write_file(file_path, new_content.encode("utf-8"))
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
        """Search for pattern in files

        Note: This requires grep/ripgrep installed in the sandbox
        """
        search_path = path or "/workspace"

        # Build grep command
        if glob:
            cmd = f"rg --json '{pattern}' --glob '{glob}' {search_path}"
        else:
            cmd = f"rg --json '{pattern}' {search_path}"

        try:
            result = self._run_async(self.provider.execute(cmd, cwd=search_path))
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
                import json

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
        """Find files matching glob pattern"""
        # Use find command to match glob pattern
        search_path = path.rstrip("/")
        cmd = f"find {search_path} -name '{pattern}'"

        try:
            result = self._run_async(self.provider.execute(cmd, cwd=search_path))
        except Exception:
            return []

        if result["exit_code"] != 0:
            return []

        files = []
        for file_path in result["stdout"].split("\n"):
            if not file_path:
                continue

            try:
                metadata = self._run_async(self.provider.get_file_metadata(file_path))
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
