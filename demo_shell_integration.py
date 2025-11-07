"""Shell tool integration for RunloopBackend.

This module provides a shell tool that allows deepagents to execute bash commands
in a remote Runloop devbox. This is essential for git operations, package management,
and any command-line interactions.

Architecture:
    User → Agent → shell_tool → RunloopBackend.exec() → Runloop API → Devbox

Key Design Decisions:
    1. Factory pattern: create_remote_shell_tool() binds the tool to a specific backend
    2. Error handling: Non-zero exit codes are surfaced clearly to the agent
    3. Type hints: Full typing for IDE support and documentation
    4. Detailed docstrings: Agent can understand tool purpose from description
"""

from typing import Annotated

from langchain_core.tools import tool

from deepagents.integrations.runloop import RunloopBackend


def create_remote_shell_tool(backend: RunloopBackend):
    """Factory function to create a shell tool bound to a specific RunloopBackend.

    This factory pattern is necessary because:
    1. Each tool needs to be bound to a specific devbox (via backend)
    2. LangChain tools are typically module-level, but we need instance-specific behavior
    3. The @tool decorator creates a new tool instance each time

    Args:
        backend: RunloopBackend instance connected to a specific devbox

    Returns:
        A LangChain tool that executes commands in the remote devbox

    Example:
        >>> backend = RunloopBackend(devbox_id="dbx_123", client=client)
        >>> shell_tool = create_remote_shell_tool(backend)
        >>> result = shell_tool.invoke({"command": "echo hello"})
        >>> print(result)
        'hello'
    """

    @tool
    def shell(
        command: Annotated[
            str,
            "The bash command to execute in the remote devbox. "
            "Use this for git operations (git status, git add, git commit, git push), "
            "GitHub CLI (gh pr create, gh issue list), "
            "package managers (npm install, pip install), "
            "file system operations (ls, cd, mkdir), "
            "and any other command-line tools available in the devbox."
        ]
    ) -> str:
        """Execute a bash command in the remote Runloop devbox.

        The devbox is a fully-featured Ubuntu environment with:
        - git (for version control)
        - gh (GitHub CLI for PR/issue management)
        - Python, Node.js, and common development tools
        - All standard Unix utilities

        Command execution details:
        - Commands run in the devbox's default working directory (/home/user)
        - Environment variables from devbox creation are available
        - Each command runs independently (no persistent shell session)
        - Use 'cd DIR && command' to run commands in specific directories

        Error handling:
        - Non-zero exit codes are reported clearly
        - stderr is included in the output for debugging
        - Long-running commands will timeout according to Runloop limits

        Args:
            command: The bash command to execute

        Returns:
            Command output (stdout) if successful, or error message with exit code

        Examples:
            # Check git status
            shell("git status")

            # Run command in specific directory
            shell("cd /home/user/repo && git add -A")

            # Chain multiple commands
            shell("git add . && git commit -m 'Update README' && git push")

            # Use gh CLI for PR
            shell("gh pr create --title 'Fix bug' --body 'Detailed description'")
        """
        # Execute command via RunloopBackend
        # backend.exec() returns tuple: (stdout: str, exit_code: int)
        stdout, exit_code = backend.exec(command)

        # Check for command failure
        if exit_code != 0:
            # Non-zero exit code indicates failure
            # Return error message that clearly shows what went wrong
            # This helps the agent understand and potentially retry with corrections
            return (
                f"❌ Command failed with exit code {exit_code}\n"
                f"Command: {command}\n"
                f"Output:\n{stdout}"
            )

        # Success case: return stdout
        # Most commands output useful information (git status, ls, etc.)
        # Some commands have no output (git add), which is fine
        return stdout if stdout else "✓ Command completed successfully (no output)"

    # Return the tool instance bound to this specific backend
    return shell


# Example usage (not executed when imported):
if __name__ == "__main__":
    # This demonstrates how to use the shell tool integration
    import os
    from runloop_api_client import Runloop

    # 1. Create Runloop client and devbox
    client = Runloop(bearer_token=os.environ["RUNLOOP_API_KEY"])
    devbox = client.devboxes.create()

    # 2. Create backend
    backend = RunloopBackend(devbox_id=devbox.id, client=client)

    # 3. Create shell tool bound to this backend
    shell_tool = create_remote_shell_tool(backend)

    # 4. Use the tool
    result = shell_tool.invoke({"command": "echo 'Hello from devbox!'"})
    print(result)  # Output: Hello from devbox!

    # 5. Cleanup
    client.devboxes.shutdown(id=devbox.id)
