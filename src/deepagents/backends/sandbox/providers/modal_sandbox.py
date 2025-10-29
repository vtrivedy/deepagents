"""Modal Sandbox provider - uses Modal's Sandbox API for persistent containers"""

import logging
import modal
import json
import base64
import asyncio
from pathlib import Path
from typing import Optional
from deepagents.backends.sandbox.types import (
    SandboxConfig,
    ExecutionResult,
    FileMetadata,
    BootstrapConfig,
)
from deepagents.backends.protocol import WriteResult, EditResult
from deepagents.backends.utils import (
    FileInfo,
    GrepMatch,
    format_content_with_line_numbers,
    perform_string_replacement,
)

logger = logging.getLogger(__name__)


class ModalSandboxFilesystem:
    """Synchronous filesystem interface for Modal Sandbox

    Provides a clean API for file operations without needing to use async/await.
    All methods are synchronous wrappers around the async provider methods.
    """

    def __init__(self, provider: "ModalSandboxProvider"):
        self.provider = provider

    def write(self, path: str, content: bytes | str) -> None:
        """Write content to a file in the sandbox

        Args:
            path: Path in the sandbox filesystem (e.g., "/workspace/file.txt")
            content: Content to write (bytes or string)
        """
        if isinstance(content, str):
            content = content.encode()
        asyncio.run(self.provider.write_file(path, content))

    def read(self, path: str) -> bytes:
        """Read content from a file in the sandbox

        Args:
            path: Path in the sandbox filesystem

        Returns:
            File content as bytes
        """
        return asyncio.run(self.provider.read_file(path))

    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        """Read text content from a file in the sandbox

        Args:
            path: Path in the sandbox filesystem
            encoding: Text encoding (default: utf-8)

        Returns:
            File content as string
        """
        return self.read(path).decode(encoding)

    def list(self, path: str, recursive: bool = False) -> list[FileMetadata]:
        """List files in a directory

        Args:
            path: Directory path in the sandbox
            recursive: If True, list files recursively

        Returns:
            List of file metadata dictionaries
        """
        return asyncio.run(self.provider.list_files(path, recursive))

    def exists(self, path: str) -> bool:
        """Check if a file or directory exists

        Args:
            path: Path in the sandbox filesystem

        Returns:
            True if the path exists
        """
        return asyncio.run(self.provider.file_exists(path))

    def delete(self, path: str) -> None:
        """Delete a file or directory

        Args:
            path: Path in the sandbox filesystem
        """
        asyncio.run(self.provider.delete_file(path))

    def mkdir(self, path: str) -> None:
        """Create a directory

        Args:
            path: Directory path in the sandbox
        """
        asyncio.run(self.provider.create_directory(path))

    def stat(self, path: str) -> FileMetadata:
        """Get file metadata

        Args:
            path: Path in the sandbox filesystem

        Returns:
            File metadata dictionary
        """
        return asyncio.run(self.provider.get_file_metadata(path))


class ModalSandboxProvider:
    """Modal Sandbox provider using persistent container per agent thread

    Key features:
    - Lazy initialization: sandbox created on first use
    - Persistent container: same container for all operations
    - Automatic cleanup: call cleanup() to terminate
    - Bootstrap support: git clone, copy files, run setup
    - Sandbox ID tracking: access via .sandbox_id property
    """

    def __init__(self, config: SandboxConfig):
        self.config = config
        self._sandbox: Optional[modal.Sandbox] = None
        self._sandbox_id: Optional[str] = None
        self._initialized = False
        self._init_lock = asyncio.Lock()  # Prevent multiple sandboxes from being created
        # Create ephemeral app - will be started with app.run() context manager
        self._app = modal.App("deepagents-sandbox")
        self._app_context = None  # Will hold the app.run() context
        self._fs: Optional[ModalSandboxFilesystem] = None

    @property
    def fs(self) -> ModalSandboxFilesystem:
        """Get synchronous filesystem interface

        Returns a filesystem object that provides clean, synchronous access
        to the sandbox filesystem without needing async/await.

        Example:
            provider = ModalSandboxProvider.from_config(config)
            fs = provider.fs
            fs.write("/workspace/file.txt", "Hello, world!")
            content = fs.read_text("/workspace/file.txt")
        """
        if self._fs is None:
            self._fs = ModalSandboxFilesystem(self)
        return self._fs

    @classmethod
    def from_config(cls, config: SandboxConfig) -> "ModalSandboxProvider":
        """Create provider from configuration"""
        return cls(config)

    @property
    def sandbox_id(self) -> str | None:
        """Get the sandbox ID for tracking"""
        return self._sandbox_id

    async def ensure_ready(self) -> None:
        """Lazily create and initialize sandbox on first use"""
        if self._initialized:
            return

        # Use lock to prevent multiple concurrent initializations
        async with self._init_lock:
            # Double-check after acquiring lock
            if self._initialized:
                return

            logger.info(f"Creating sandbox container (image={self.config.image})...")

            # Start the ephemeral app context
            self._app_context = self._app.run()
            self._app_context.__enter__()

            # Convert image string to Modal Image object
            modal_image = modal.Image.from_registry(self.config.image)

            # Create the sandbox within the ephemeral app
            self._sandbox = modal.Sandbox.create(
                app=self._app,
                image=modal_image,
                timeout=self.config.timeout_seconds,
                cpu=self.config.cpu_count,
                memory=self.config.memory_mb,
            )

            # Get sandbox ID for tracking
            self._sandbox_id = self._sandbox.object_id
            logger.info(f"Sandbox ready: {self._sandbox_id}")

            # Setup working directory
            # Note: Don't use workdir as cwd when creating it (it doesn't exist yet!)
            result = await self._exec_helper(f"mkdir -p {self.config.workdir}", cwd="/")
            if result["exit_code"] != 0:
                raise RuntimeError(f"Failed to create workdir: {result['stderr']}")
            logger.info(f"Created workdir: {self.config.workdir}")

            # Run bootstrap if configured
            if self.config.bootstrap:
                logger.info("Running bootstrap...")
                await self._bootstrap(self.config.bootstrap)
                logger.info("Bootstrap complete")

            self._initialized = True

    async def _bootstrap(self, bootstrap: BootstrapConfig) -> None:
        """Bootstrap the sandbox environment"""

        # Clone git repository
        if bootstrap.git_repo:
            logger.info(f"Cloning {bootstrap.git_repo}...")
            branch_flag = f"-b {bootstrap.git_branch}" if bootstrap.git_branch else ""
            clone_cmd = f"git clone {branch_flag} {bootstrap.git_repo} {bootstrap.workdir}/repo"
            result = await self._exec_helper(clone_cmd)
            if result["exit_code"] != 0:
                raise RuntimeError(f"Git clone failed: {result['stderr']}")

        # Copy local files
        if bootstrap.local_files:
            logger.info(f"Copying {len(bootstrap.local_files)} files...")
            for local_path, sandbox_path in bootstrap.local_files.items():
                local_file = Path(local_path)
                if not local_file.exists():
                    raise FileNotFoundError(f"Local file not found: {local_path}")

                content = local_file.read_bytes()
                await self.write_file(sandbox_path, content)

        # Run setup script
        if bootstrap.setup_script:
            logger.info("Running setup script...")
            result = await self._exec_helper(
                bootstrap.setup_script,
                cwd=bootstrap.workdir
            )
            if result["exit_code"] != 0:
                raise RuntimeError(f"Setup script failed: {result['stderr']}")

    async def _exec_helper(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """Helper to execute command (doesn't call ensure_ready to avoid recursion)"""
        if not self._sandbox:
            raise RuntimeError("Sandbox not initialized")

        # Merge environment variables
        full_env = {}
        if self.config.env_vars:
            full_env.update(self.config.env_vars)
        if env:
            full_env.update(env)

        # Execute in sandbox
        process = self._sandbox.exec(
            "bash",
            "-c",
            command,
            workdir=cwd or self.config.workdir,
            env=full_env if full_env else None,
        )

        # Wait for process to complete before accessing returncode
        process.wait()

        return {
            "stdout": process.stdout.read(),
            "stderr": process.stderr.read(),
            "exit_code": process.returncode,
        }

    async def _execute_async_internal(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """Execute a shell command in the sandbox (internal async method)"""
        await self.ensure_ready()
        return await self._exec_helper(command, cwd, env)

    async def read_file(self, path: str) -> bytes:
        """Read file from sandbox using cat and base64 encoding"""
        await self.ensure_ready()

        # Use base64 to handle binary files correctly
        result = await self._exec_helper(f"base64 {path}")

        if result["exit_code"] != 0:
            raise FileNotFoundError(f"File not found or cannot be read: {path}")

        # Decode base64
        return base64.b64decode(result["stdout"])

    async def write_file(self, path: str, content: bytes) -> None:
        """Write file to sandbox using base64 encoding"""
        await self.ensure_ready()

        # Encode content as base64 for safe shell transfer
        encoded = base64.b64encode(content).decode("ascii")

        # Create directory if needed
        dir_path = str(Path(path).parent)
        await self._exec_helper(f"mkdir -p {dir_path}", cwd="/")

        # Write using base64 decode
        result = await self._exec_helper(
            f"echo '{encoded}' | base64 -d > {path}"
        )

        if result["exit_code"] != 0:
            raise RuntimeError(f"Failed to write file: {result['stderr']}")

    async def list_files(
        self,
        path: str,
        recursive: bool = False,
    ) -> list[FileMetadata]:
        """List files in directory"""
        await self.ensure_ready()

        # Build find command
        if recursive:
            cmd = f"find {path} -type f -o -type d"
        else:
            cmd = f"find {path} -maxdepth 1 -type f -o -type d"

        result = await self._exec_helper(cmd)

        if result["exit_code"] != 0:
            return []

        files = []
        for line in result["stdout"].strip().split("\n"):
            if not line or line == path:
                continue

            # Get file metadata
            stat_cmd = f"stat -c '%s %Y %F' {line}"
            stat_result = await self._exec_helper(stat_cmd)

            if stat_result["exit_code"] == 0:
                parts = stat_result["stdout"].strip().split(maxsplit=2)
                if len(parts) >= 3:
                    size = int(parts[0])
                    mtime = parts[1]
                    ftype = parts[2]

                    files.append({
                        "path": line,
                        "size": size,
                        "is_dir": "directory" in ftype.lower(),
                        "modified_at": mtime,
                    })

        return files

    async def delete_file(self, path: str) -> None:
        """Delete a file or directory"""
        await self.ensure_ready()

        result = await self._exec_helper(f"rm -rf {path}")

        if result["exit_code"] != 0:
            raise RuntimeError(f"Failed to delete: {result['stderr']}")

    async def file_exists(self, path: str) -> bool:
        """Check if file exists"""
        await self.ensure_ready()

        result = await self._exec_helper(f"test -e {path}")
        return result["exit_code"] == 0

    async def create_directory(self, path: str) -> None:
        """Create directory"""
        await self.ensure_ready()

        result = await self._exec_helper(f"mkdir -p {path}")

        if result["exit_code"] != 0:
            raise RuntimeError(f"Failed to create directory: {result['stderr']}")

    async def get_file_metadata(self, path: str) -> FileMetadata:
        """Get file metadata"""
        await self.ensure_ready()

        stat_cmd = f"stat -c '%s %Y %F' {path}"
        result = await self._exec_helper(stat_cmd)

        if result["exit_code"] != 0:
            raise FileNotFoundError(f"File not found: {path}")

        parts = result["stdout"].strip().split(maxsplit=2)
        if len(parts) < 3:
            raise RuntimeError(f"Failed to parse stat output: {result['stdout']}")

        size = int(parts[0])
        mtime = parts[1]
        ftype = parts[2]

        return {
            "path": path,
            "size": size,
            "is_dir": "directory" in ftype.lower(),
            "modified_at": mtime,
        }

    async def cleanup(self) -> None:
        """Terminate sandbox and clean up all resources"""
        if self._sandbox:
            logger.info(f"Terminating sandbox {self._sandbox_id}...")
            try:
                self._sandbox.terminate()
                logger.info("Sandbox terminated")
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
            finally:
                self._sandbox = None
                self._sandbox_id = None
                self._initialized = False

        # Exit the ephemeral app context
        if self._app_context:
            try:
                self._app_context.__exit__(None, None, None)
                logger.info("Ephemeral app stopped")
            except Exception as e:
                logger.error(f"Error stopping app: {e}")
            finally:
                self._app_context = None

    def __enter__(self) -> "ModalSandboxProvider":
        """Context manager entry - enables `with provider:` syntax"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically cleanup sandbox"""
        import asyncio
        asyncio.run(self.cleanup())
        return False

    # ============================================================================
    # BackendProtocol Implementation
    # ============================================================================
    # These methods implement the BackendProtocol interface, allowing Modal
    # sandbox to be used as a backend for filesystem operations.

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
        """Write file to sandbox (BackendProtocol method)"""
        try:
            self._run_async(
                self.write_file(file_path, content.encode("utf-8"))
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
        """Upload file to sandbox (convenience method)

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
            provider.upload_file(
                path="/workspace/data.csv",
                local_path="./local_data.csv"
            )

            # Upload from bytes
            provider.upload_file(
                path="/workspace/file.bin",
                content=b"binary data"
            )

            # Upload from string
            provider.upload_file(
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
        """Edit file in sandbox using string replacement (BackendProtocol method)"""
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

        Note: This requires grep/ripgrep installed in the sandbox
        """
        search_path = path or "/workspace"

        # Build grep command
        if glob:
            cmd = f"rg --json '{pattern}' --glob '{glob}' {search_path}"
        else:
            cmd = f"rg --json '{pattern}' {search_path}"

        try:
            result = self._run_async(self._execute_async_internal(cmd, cwd=search_path))
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
            result = self._run_async(self._execute_async_internal(cmd, cwd=search_path))
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

    def execute(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> dict[str, any]:
        """Execute command in sandbox (sync wrapper for bash tool)

        This is a sync wrapper around the async execute() method, allowing
        the bash tool to call it synchronously.

        Args:
            command: Shell command to execute
            cwd: Working directory for command execution
            env: Environment variables

        Returns:
            ExecutionResult dict with stdout, stderr, exit_code
        """
        return self._run_async(self._execute_async_internal(command, cwd, env))


    def __del__(self):
        """Cleanup on deletion (safety net, but explicit cleanup() is preferred)"""
        if self._sandbox and self._initialized:
            logger.warning(f"Sandbox {self._sandbox_id} not explicitly cleaned up! Call cleanup() to avoid resource leaks.")
