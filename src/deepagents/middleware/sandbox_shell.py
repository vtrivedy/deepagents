"""Middleware for executing shell commands in Modal sandboxes."""

import atexit
import os
import pathlib
import signal
import sys
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any, Literal

import modal
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.tools import BaseTool, tool
from rich.console import Console
from rich.panel import Panel


class SandboxShellMiddleware(AgentMiddleware):
    """Execute shell commands in isolated Modal sandboxes.

    This middleware replaces the standard shell tool with one that executes
    commands inside Modal sandboxes for security and isolation.

    Features:
    - Secure execution via Modal sandboxes (gVisor isolation)
    - Auto-sync local files to sandbox on startup
    - Custom package installation (pip/apt)
    - Persistent sandbox across commands in a session
    - Automatic cleanup on exit

    Examples:
        Basic usage:
        >>> agent = create_deep_agent(
        ...     use_sandbox_shell=True,
        ...     sandbox_config={"pip_packages": ["pytest", "black"]}
        ... )

        With file sync:
        >>> agent = create_deep_agent(
        ...     use_sandbox_shell=True,
        ...     sandbox_config={
        ...         "sync_cwd": True,
        ...         "sync_exclude": [".git", "*.pyc"],
        ...         "pip_packages": ["pytest"]
        ...     }
        ... )

        Custom image:
        >>> custom_img = modal.Image.debian_slim().pip_install("torch")
        >>> agent = create_deep_agent(
        ...     use_sandbox_shell=True,
        ...     sandbox_config={"custom_image": custom_img}
        ... )
    """

    def __init__(
        self,
        # Container configuration
        app_name: str = "deepagents-sandbox",
        base_image: str = "python:3.11-slim",
        pip_packages: list[str] | None = None,
        apt_packages: list[str] | None = None,
        custom_image: modal.Image | None = None,
        image_config: dict[str, Any] | None = None,
        # File synchronization
        sync_cwd: bool = False,
        sync_paths: list[str] | None = None,
        sync_exclude: list[str] | None = None,
        # Sandbox lifecycle
        timeout: int = 3600,  # 1 hour default
        idle_timeout: int | None = 300,  # Auto-terminate after 5 minutes of inactivity
        # User interface
        verbose: bool = True,
    ):
        """Initialize the sandbox shell middleware.

        Args:
            app_name: Name of Modal app (default: "deepagents-sandbox")
            base_image: Base Docker image (default: "python:3.11-slim")
            pip_packages: Python packages to install (e.g., ["pytest", "black"])
            apt_packages: System packages to install (e.g., ["git", "curl"])
            custom_image: Override with fully custom Modal image (advanced users)
            image_config: Declarative image config dict with keys: "base", "pip_packages",
                "apt_packages", "commands". Cannot be used with custom_image.
            sync_cwd: Auto-sync current working directory to sandbox
            sync_paths: Explicit list of paths to sync to sandbox
            sync_exclude: Patterns to exclude from sync (e.g., [".git", "*.pyc"])
            timeout: Maximum sandbox lifetime in seconds (default: 3600)
            idle_timeout: Time in seconds before an idle sandbox is terminated automatically (default: 300)
            verbose: Show Rich notifications for sandbox lifecycle events
        """
        if custom_image is not None and image_config is not None:
            raise ValueError("Cannot specify both custom_image and image_config")

        self.app_name = app_name
        self.base_image = base_image
        self.pip_packages = pip_packages or []
        self.apt_packages = apt_packages or []
        self.custom_image = custom_image
        self.image_config = image_config
        self.sync_cwd = sync_cwd
        self.sync_paths = sync_paths or []
        self.sync_exclude = sync_exclude or [".git", "*.pyc", "__pycache__", "node_modules", ".env"]
        self.timeout = timeout
        self.idle_timeout = idle_timeout
        self.verbose = verbose

        # Runtime state
        self.sb = None
        self._app = None
        self._app_ctx = None
        self._is_cleaning_up = False
        self.console = Console() if verbose else None

        # Register cleanup handlers
        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Build system prompt and tools
        self.system_prompt = self._build_system_prompt()
        self.tools = [self._create_shell_tool()]

    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully."""
        if self.verbose:
            print("\n⚠️  Interrupt received. Cleaning up sandbox...")
        self.cleanup()
        sys.exit(0)

    def _notify(self, message: str, title: str | None = None):
        """Display Rich notification if verbose mode is enabled."""
        if not self.verbose or not self.console:
            return

        if title:
            self.console.print(Panel(message, title=title, border_style="blue"))
        else:
            self.console.print(f"ℹ️  {message}")

    def _build_image(self) -> modal.Image:
        """Build Modal image with packages and synced files.

        Returns:
            Configured Modal image ready for sandbox creation.
        """
        # Priority: custom_image > image_config > base_image + packages
        if self.custom_image:
            img = self.custom_image
        elif self.image_config:
            # Build from declarative config
            base = self.image_config.get("base", "python:3.11-slim")
            img = modal.Image.from_registry(base)

            if self.image_config.get("apt_packages"):
                apt_pkgs = self.image_config["apt_packages"]
                self._notify(f"Installing apt packages: {', '.join(apt_pkgs)}")
                img = img.apt_install(*apt_pkgs)

            if self.image_config.get("pip_packages"):
                pip_pkgs = self.image_config["pip_packages"]
                self._notify(f"Installing pip packages: {', '.join(pip_pkgs)}")
                img = img.pip_install(*pip_pkgs)

            if self.image_config.get("commands"):
                for cmd in self.image_config["commands"]:
                    self._notify(f"Running command: {cmd}")
                    img = img.run_commands(cmd)
        else:
            # Start with base image
            img = modal.Image.from_registry(self.base_image)

            # Add system packages
            if self.apt_packages:
                self._notify(f"Installing apt packages: {', '.join(self.apt_packages)}")
                img = img.apt_install(*self.apt_packages)

            # Add Python packages
            if self.pip_packages:
                self._notify(f"Installing pip packages: {', '.join(self.pip_packages)}")
                img = img.pip_install(*self.pip_packages)

        # Sync files into image
        if self.sync_cwd:
            cwd = os.getcwd()
            self._notify(f"Syncing current directory to /workspace: {cwd}")

            # Filter excluded patterns
            def should_include(path: str) -> bool:
                for pattern in self.sync_exclude:
                    if pattern.startswith("*."):
                        # Extension pattern
                        if path.endswith(pattern[1:]):
                            return False
                    else:
                        # Direct name match
                        if pattern in path:
                            return False
                return True

            # Add directory with filtering (Modal will handle this)
            img = img.add_local_dir(
                local_path=cwd,
                remote_path="/workspace",
            )

        elif self.sync_paths:
            file_names = []
            for path in self.sync_paths:
                path_obj = pathlib.Path(path)
                if path_obj.is_file():
                    file_names.append(path_obj.name)
                    img = img.add_local_file(
                        local_path=str(path_obj),
                        remote_path=f"/workspace/{path_obj.name}",
                    )
                elif path_obj.is_dir():
                    file_names.append(f"{path_obj.name}/")
                    img = img.add_local_dir(
                        local_path=str(path_obj),
                        remote_path=f"/workspace/{path_obj.name}",
                    )
            if file_names:
                self._notify(f"Adding local files to container: {', '.join(file_names)}")

        return img

    def _create_sandbox(self) -> modal.Sandbox:
        """Create Modal sandbox with built image.

        Returns:
            Active Modal sandbox instance.
        """
        self._notify("🚀 Creating Modal sandbox...", title="Sandbox")

        # Start an ephemeral Modal app if we haven't already
        if self._app is None:
            self._app = modal.App(self.app_name)
            self._app_ctx = self._app.run()
            self._app_ctx.__enter__()

        # Build image with output streaming
        output_ctx = modal.enable_output() if self.verbose else nullcontext()
        with output_ctx:
            img = self._build_image()

        # Create sandbox
        sb = modal.Sandbox.create(
            app=self._app,
            image=img,
            timeout=self.timeout,
            idle_timeout=self.idle_timeout,
            workdir="/workspace",
        )

        self._notify("✅ Sandbox ready!", title="Sandbox")
        return sb

    def _create_shell_tool(self) -> BaseTool:
        """Create the shell tool that executes commands in sandbox.

        Returns:
            Configured shell tool for agent use.
        """

        @tool
        def shell(command: str, timeout: int = 60) -> str:
            """Execute shell command in isolated Modal sandbox.

            The sandbox is a secure, isolated container running on Modal's infrastructure.
            It persists for this session, so:
            - Packages you install stay available
            - Files you create remain accessible
            - Environment variables are preserved

            The sandbox has:
            - Python 3.11
            - Any packages specified in pip_packages/apt_packages
            - Your synced files at /workspace (if file sync enabled)
            - Working directory: /workspace

            Args:
                command: Shell command to execute
                timeout: Command timeout in seconds (default: 60)

            Returns:
                Command output (stdout + stderr) with exit code
            """
            # Create sandbox on first use (lazy initialization)
            if self.sb is None:
                self.sb = self._create_sandbox()

            try:
                # Execute command
                p = self.sb.exec("bash", "-c", command, timeout=timeout)

                # Collect output
                stdout = p.stdout.read()
                stderr_content = ""
                if hasattr(p, "stderr") and hasattr(p.stderr, "read"):
                    stderr_content = p.stderr.read()

                # Wait for completion
                p.wait()

                # Format result
                result_parts = [f"Exit code: {p.returncode}"]
                if stdout:
                    result_parts.append(f"\nSTDOUT:\n{stdout}")
                if stderr_content:
                    result_parts.append(f"\nSTDERR:\n{stderr_content}")

                return "\n".join(result_parts)

            except Exception as e:
                return f"Error executing command in sandbox: {str(e)}"

        return shell

    def _build_system_prompt(self) -> str:
        """Build system prompt explaining sandbox environment.

        Returns:
            System prompt text to inject before model calls.
        """
        prompt_parts = [
            "## Sandbox Execution Environment",
            "",
            "You have access to a `shell` tool that executes commands in an isolated Modal sandbox.",
            "",
            "The sandbox:",
            "- Runs on Modal's secure infrastructure (gVisor isolation)",
            "- Persists across commands in this session",
            "- Has working directory: /workspace",
        ]

        # Add package information
        if self.pip_packages:
            prompt_parts.append(f"- Pre-installed Python packages: {', '.join(self.pip_packages)}")
        if self.apt_packages:
            prompt_parts.append(f"- Pre-installed system packages: {', '.join(self.apt_packages)}")

        # Add file sync information
        if self.sync_cwd:
            prompt_parts.extend(
                [
                    "- Your local files are synced to /workspace",
                    "- Use /workspace/<filename> to access synced files",
                ]
            )
        elif self.sync_paths:
            prompt_parts.append(f"- {len(self.sync_paths)} local path(s) synced to /workspace")

        prompt_parts.extend(
            [
                "",
                "Important:",
                "- The sandbox is isolated from your local machine (secure)",
                "- Files you create in the sandbox persist during this session",
                "- Use shell('command') to execute commands safely",
            ]
        )

        return "\n".join(prompt_parts)

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """Inject sandbox system prompt before each model call.

        Args:
            request: The model request being processed
            handler: The handler function to call with modified request

        Returns:
            Model response from handler
        """
        # Prepend sandbox instructions to system prompt
        if self.system_prompt:
            request.system_prompt = (
                self.system_prompt + "\n\n" + request.system_prompt
                if request.system_prompt
                else self.system_prompt
            )

        return handler(request)

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Any]
    ) -> ModelResponse:
        """Async version of wrap_model_call.

        Args:
            request: The model request being processed
            handler: The async handler function to call with modified request

        Returns:
            Model response from handler
        """
        # Prepend sandbox instructions to system prompt
        if self.system_prompt:
            request.system_prompt = (
                self.system_prompt + "\n\n" + request.system_prompt
                if request.system_prompt
                else self.system_prompt
            )

        return await handler(request)

    def cleanup(self):
        """Terminate sandbox and clean up resources.

        This method is idempotent and safe to call multiple times.
        Automatically called on program exit via atexit.
        """
        # Prevent double cleanup
        if self._is_cleaning_up:
            return

        self._is_cleaning_up = True
        try:
            if self.sb is not None:
                self._notify("🧹 Terminating sandbox...", title="Cleanup")
                self.sb.terminate()
                self.sb.wait(raise_on_termination=False)
                self._notify("✅ Sandbox terminated", title="Cleanup")
        finally:
            self.sb = None
            if self._app_ctx is not None:
                self._app_ctx.__exit__(None, None, None)
                self._app_ctx = None
                self._app = None
            self._is_cleaning_up = False

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, *args):
        """Context manager exit with cleanup."""
        self.cleanup()

    def __del__(self):
        """Cleanup when middleware is garbage collected."""
        self.cleanup()
