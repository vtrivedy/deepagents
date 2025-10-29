#!/usr/bin/env python3
import sys

# Check for CLI dependencies before importing them
def check_cli_dependencies():
    """Check if CLI optional dependencies are installed."""
    missing = []
    
    try:
        import rich
    except ImportError:
        missing.append("rich")
    
    try:
        import requests
    except ImportError:
        missing.append("requests")
    
    try:
        import dotenv
    except ImportError:
        missing.append("python-dotenv")
    
    try:
        import tavily
    except ImportError:
        missing.append("tavily-python")

    try:
        import prompt_toolkit
    except ImportError:
        missing.append("prompt-toolkit")

    if missing:
        print("\n❌ Missing required CLI dependencies!")
        print(f"\nThe following packages are required to use the deepagents CLI:")
        for pkg in missing:
            print(f"  - {pkg}")
        print(f"\nPlease install them with:")
        print(f"  pip install deepagents[cli]")
        print(f"\nOr install all dependencies:")
        print(f"  pip install 'deepagents[cli]'")
        sys.exit(1)

check_cli_dependencies()

import argparse
import asyncio
import os
import subprocess
import platform
import requests
from typing import Dict, Any, Union, Literal
from pathlib import Path

from tavily import TavilyClient
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langchain.agents.middleware import ShellToolMiddleware, HostExecutionPolicy, InterruptOnConfig

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends import CompositeBackend
from deepagents.middleware.agent_memory import AgentMemoryMiddleware
from pathlib import Path
import shutil
from rich import box

import dotenv
import re
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.completion import Completer, PathCompleter, WordCompleter, merge_completers, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.enums import EditingMode

dotenv.load_dotenv()

COLORS = {
    "primary": "#10b981",
    "dim": "#6b7280",
    "user": "#ffffff",
    "agent": "#10b981",
    "thinking": "#34d399",
    "tool": "#fbbf24",
}

DEEP_AGENTS_ASCII = """
 ██████╗  ███████╗ ███████╗ ██████╗
 ██╔══██╗ ██╔════╝ ██╔════╝ ██╔══██╗
 ██║  ██║ █████╗   █████╗   ██████╔╝
 ██║  ██║ ██╔══╝   ██╔══╝   ██╔═══╝
 ██████╔╝ ███████╗ ███████╗ ██║
 ╚═════╝  ╚══════╝ ╚══════╝ ╚═╝

  █████╗   ██████╗  ███████╗ ███╗   ██╗ ████████╗ ███████╗
 ██╔══██╗ ██╔════╝  ██╔════╝ ████╗  ██║ ╚══██╔══╝ ██╔════╝
 ███████║ ██║  ███╗ █████╗   ██╔██╗ ██║    ██║    ███████╗
 ██╔══██║ ██║   ██║ ██╔══╝   ██║╚██╗██║    ██║    ╚════██║
 ██║  ██║ ╚██████╔╝ ███████╗ ██║ ╚████║    ██║    ███████║
 ╚═╝  ╚═╝  ╚═════╝  ╚══════╝ ╚═╝  ╚═══╝    ╚═╝    ╚══════╝
"""

console = Console()

tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY")) if os.environ.get("TAVILY_API_KEY") else None


def http_request(
    url: str,
    method: str = "GET",
    headers: Dict[str, str] = None,
    data: Union[str, Dict] = None,
    params: Dict[str, str] = None,
    timeout: int = 30,
) -> Dict[str, Any]:
    """
    Make HTTP requests to APIs and web services.

    Args:
        url: Target URL
        method: HTTP method (GET, POST, PUT, DELETE, etc.)
        headers: HTTP headers to include
        data: Request body data (string or dict)
        params: URL query parameters
        timeout: Request timeout in seconds

    Returns:
        Dictionary with response data including status, headers, and content
    """
    try:
        kwargs = {"url": url, "method": method.upper(), "timeout": timeout}

        if headers:
            kwargs["headers"] = headers
        if params:
            kwargs["params"] = params
        if data:
            if isinstance(data, dict):
                kwargs["json"] = data
            else:
                kwargs["data"] = data

        response = requests.request(**kwargs)

        try:
            content = response.json()
        except:
            content = response.text

        return {
            "success": response.status_code < 400,
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "content": content,
            "url": response.url,
        }

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "status_code": 0,
            "headers": {},
            "content": f"Request timed out after {timeout} seconds",
            "url": url,
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "status_code": 0,
            "headers": {},
            "content": f"Request error: {str(e)}",
            "url": url,
        }
    except Exception as e:
        return {
            "success": False,
            "status_code": 0,
            "headers": {},
            "content": f"Error making request: {str(e)}",
            "url": url,
        }


def web_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Search the web using Tavily for programming-related information."""
    if tavily_client is None:
        return {
            "error": "Tavily API key not configured. Please set TAVILY_API_KEY environment variable.",
            "query": query
        }
    
    try:
        search_docs = tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
        )
        return search_docs
    except Exception as e:
        return {
            "error": f"Web search error: {str(e)}",
            "query": query
        }


def get_default_coding_instructions() -> str:
    """Get the default coding agent instructions.

    These are the immutable base instructions that cannot be modified by the agent.
    Long-term memory (agent.md) is handled separately by the middleware.
    """
    default_prompt_path = Path(__file__).parent / "default_agent_prompt.md"
    return default_prompt_path.read_text()


def create_model():
    """Create the appropriate model based on available API keys.

    Returns:
        ChatModel instance (OpenAI or Anthropic)

    Raises:
        SystemExit if no API key is configured
    """
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    if openai_key:
        from langchain_openai import ChatOpenAI
        model_name = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
        console.print(f"[dim]Using OpenAI model: {model_name}[/dim]")
        return ChatOpenAI(
            model=model_name,
            temperature=0.7,
        )
    elif anthropic_key:
        from langchain_anthropic import ChatAnthropic
        model_name = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
        console.print(f"[dim]Using Anthropic model: {model_name}[/dim]")
        return ChatAnthropic(
            model_name=model_name,
            max_tokens=20000,
        )
    else:
        console.print("[bold red]Error:[/bold red] No API key configured.")
        console.print("\nPlease set one of the following environment variables:")
        console.print("  - OPENAI_API_KEY     (for OpenAI models like gpt-5-mini)")
        console.print("  - ANTHROPIC_API_KEY  (for Claude models)")
        console.print("\nExample:")
        console.print("  export OPENAI_API_KEY=your_api_key_here")
        console.print("\nOr add it to your .env file.")
        sys.exit(1)


config = {"recursion_limit": 1000}

MAX_ARG_LENGTH = 150


def truncate_value(value: str, max_length: int = MAX_ARG_LENGTH) -> str:
    """Truncate a string value if it exceeds max_length."""
    if len(value) > max_length:
        return value[:max_length] + "..."
    return value


class TokenTracker:
    """Track token usage across the conversation."""

    def __init__(self):
        self.session_input = 0
        self.session_output = 0
        self.last_input = 0
        self.last_output = 0

    def add(self, input_tokens: int, output_tokens: int):
        """Add tokens from a response."""
        self.session_input += input_tokens
        self.session_output += output_tokens
        self.last_input = input_tokens
        self.last_output = output_tokens

    def display_last(self):
        """Display tokens for the last response."""
        # Only show output tokens generated in this turn
        if self.last_output:
            if self.last_output >= 1000:
                console.print(f"  {self.last_output:,} tokens", style="dim")

    def display_session(self):
        """Display cumulative session tokens."""
        total = self.session_input + self.session_output
        console.print(f"\n[bold]Session Token Usage:[/bold]", style=COLORS["primary"])
        console.print(f"  Input:  {self.session_input:,} tokens", style=COLORS["dim"])
        console.print(f"  Output: {self.session_output:,} tokens", style=COLORS["dim"])
        console.print(f"  Total:  {total:,} tokens\n", style=COLORS["dim"])


class FilePathCompleter(Completer):
    """File path completer that triggers on @ symbol."""

    def __init__(self):
        self.path_completer = PathCompleter(expanduser=True)

    def get_completions(self, document, complete_event):
        """Get file path completions when @ is detected."""
        text = document.text_before_cursor

        # Check if we're after an @ symbol
        if '@' in text:
            # Get the part after the last @
            parts = text.split('@')
            if len(parts) >= 2:
                after_at = parts[-1]
                # Create a document for just the path part
                path_doc = Document(after_at, len(after_at))

                # Get completions from PathCompleter
                for completion in self.path_completer.get_completions(path_doc, complete_event):
                    # PathCompleter already gives us the correct start_position
                    # relative to the path_doc, which is what we want
                    yield Completion(
                        text=completion.text,
                        start_position=completion.start_position,
                        display=completion.display,
                        display_meta=completion.display_meta,
                        style=completion.style,
                    )


COMMANDS = {
    'clear': 'Clear screen and reset conversation',
    'help': 'Show help information',
    'tokens': 'Show token usage for current session',
    'quit': 'Exit the CLI',
    'exit': 'Exit the CLI',
}


class CommandCompleter(Completer):
    """Command completer for / commands."""

    def __init__(self):
        self.word_completer = WordCompleter(
            list(COMMANDS.keys()),
            meta_dict=COMMANDS,
            sentence=True,
            ignore_case=True,
        )

    def get_completions(self, document, complete_event):
        """Get command completions when / is at the start."""
        text = document.text

        # Only complete if line starts with /
        if text.startswith('/'):
            # Remove / for word completion
            cmd_text = text[1:]
            adjusted_doc = Document(
                cmd_text,
                document.cursor_position - 1 if document.cursor_position > 0 else 0
            )

            for completion in self.word_completer.get_completions(adjusted_doc, complete_event):
                yield completion


# Common bash commands for autocomplete (only universally available commands)
COMMON_BASH_COMMANDS = {
    'ls': 'List directory contents',
    'ls -la': 'List all files with details',
    'cd': 'Change directory',
    'pwd': 'Print working directory',
    'cat': 'Display file contents',
    'grep': 'Search text patterns',
    'find': 'Find files',
    'mkdir': 'Make directory',
    'rm': 'Remove file',
    'cp': 'Copy file',
    'mv': 'Move/rename file',
    'echo': 'Print text',
    'touch': 'Create empty file',
    'head': 'Show first lines',
    'tail': 'Show last lines',
    'wc': 'Count lines/words',
    'chmod': 'Change permissions',
}


class BashCompleter(Completer):
    """Bash command completer for ! commands."""

    def __init__(self):
        self.word_completer = WordCompleter(
            list(COMMON_BASH_COMMANDS.keys()),
            meta_dict=COMMON_BASH_COMMANDS,
            sentence=True,
            ignore_case=True,
        )

    def get_completions(self, document, complete_event):
        """Get bash command completions when ! is at the start."""
        text = document.text

        # Only complete if line starts with !
        if text.startswith('!'):
            # Remove ! for word completion
            cmd_text = text[1:]
            adjusted_doc = Document(
                cmd_text,
                document.cursor_position - 1 if document.cursor_position > 0 else 0
            )

            for completion in self.word_completer.get_completions(adjusted_doc, complete_event):
                yield completion


def parse_file_mentions(text: str) -> tuple[str, list[Path]]:
    """Extract @file mentions and return cleaned text with resolved file paths."""
    pattern = r'@((?:[^\s@]|(?<=\\)\s)+)'  # Match @filename, allowing escaped spaces
    matches = re.findall(pattern, text)

    files = []
    for match in matches:
        # Remove escape characters
        clean_path = match.replace('\\ ', ' ')
        path = Path(clean_path).expanduser()

        # Try to resolve relative to cwd
        if not path.is_absolute():
            path = Path.cwd() / path

        try:
            path = path.resolve()
            if path.exists() and path.is_file():
                files.append(path)
            else:
                console.print(f"[yellow]Warning: File not found: {match}[/yellow]")
        except Exception as e:
            console.print(f"[yellow]Warning: Invalid path {match}: {e}[/yellow]")

    return text, files


def show_interactive_help():
    """Show available commands during interactive session."""
    console.print()
    console.print("[bold]Interactive Commands:[/bold]", style=COLORS["primary"])
    console.print()

    for cmd, desc in COMMANDS.items():
        console.print(f"  /{cmd:<12} {desc}", style=COLORS["dim"])

    console.print()
    console.print("[bold]Editing Features:[/bold]", style=COLORS["primary"])
    console.print("  Enter           Submit your message", style=COLORS["dim"])
    console.print("  Alt+Enter       Insert newline (Option+Enter on Mac, or ESC then Enter)", style=COLORS["dim"])
    console.print("  Ctrl+E          Open in external editor (nano by default)", style=COLORS["dim"])
    console.print("  Arrow keys      Navigate input and history", style=COLORS["dim"])
    console.print("  Ctrl+C          Cancel current input", style=COLORS["dim"])
    console.print()
    console.print("[bold]Special Features:[/bold]", style=COLORS["primary"])
    console.print("  @filename       Type @ to auto-complete files and inject content", style=COLORS["dim"])
    console.print("  /command        Type / to see available commands", style=COLORS["dim"])
    console.print("  !command        Type ! to run bash commands (e.g., !ls, !git status)", style=COLORS["dim"])
    console.print("                  Completions appear automatically as you type", style=COLORS["dim"])
    console.print()


def handle_command(command: str, agent, token_tracker: TokenTracker) -> str | bool:
    """Handle slash commands. Returns 'exit' to exit, True if handled, False to pass to agent."""
    cmd = command.lower().strip().lstrip('/')

    if cmd in ['quit', 'exit', 'q']:
        return 'exit'

    elif cmd == 'clear':
        # Reset agent conversation state
        from langgraph.checkpoint.memory import InMemorySaver
        agent.checkpointer = InMemorySaver()

        # Clear screen and show fresh UI
        console.clear()
        console.print(DEEP_AGENTS_ASCII, style=f"bold {COLORS['primary']}")
        console.print()
        console.print("... Fresh start! Screen cleared and conversation reset.", style=COLORS["agent"])
        console.print()
        return True

    elif cmd == 'help':
        show_interactive_help()
        return True

    elif cmd == 'tokens':
        token_tracker.display_session()
        return True

    else:
        console.print()
        console.print(f"[yellow]Unknown command: /{cmd}[/yellow]")
        console.print(f"[dim]Type /help for available commands.[/dim]")
        console.print()
        return True

    return False


def execute_bash_command(command: str) -> bool:
    """Execute a bash command and display output. Returns True if handled."""
    cmd = command.strip().lstrip('!')

    if not cmd:
        return True

    try:
        console.print()
        console.print(f"[dim]$ {cmd}[/dim]")

        # Execute the command
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=Path.cwd()
        )

        # Display output
        if result.stdout:
            console.print(result.stdout, style=COLORS["dim"])
        if result.stderr:
            console.print(result.stderr, style="red")

        # Show return code if non-zero
        if result.returncode != 0:
            console.print(f"[dim]Exit code: {result.returncode}[/dim]")

        console.print()
        return True

    except subprocess.TimeoutExpired:
        console.print("[red]Command timed out after 30 seconds[/red]")
        console.print()
        return True
    except Exception as e:
        console.print(f"[red]Error executing command: {e}[/red]")
        console.print()
        return True


def create_prompt_session(assistant_id: str) -> PromptSession:
    """Create a configured PromptSession with all features."""

    # Set default editor if not already set
    if 'EDITOR' not in os.environ:
        os.environ['EDITOR'] = 'nano'

    # Create key bindings
    kb = KeyBindings()

    # Bind regular Enter to submit (intuitive behavior)
    @kb.add('enter')
    def _(event):
        """Enter submits the input, unless completion menu is active."""
        buffer = event.current_buffer

        # If completion menu is showing, apply the current completion
        if buffer.complete_state:
            # Get the current completion (the highlighted one)
            current_completion = buffer.complete_state.current_completion

            # If no completion is selected (user hasn't navigated), auto-select the first one
            if not current_completion:
                completions = buffer.complete_state.completions
                if completions:
                    current_completion = completions[0]

            if current_completion:
                # Apply the completion
                buffer.apply_completion(current_completion)
            else:
                # No completions available, close menu
                buffer.complete_state = None
        else:
            # Don't submit if buffer is empty or only whitespace
            if buffer.text.strip():
                # Normal submit
                buffer.validate_and_handle()
            # If empty, do nothing (don't submit)

    # Alt+Enter for newlines (press ESC then Enter, or Option+Enter on Mac)
    @kb.add('escape', 'enter')
    def _(event):
        """Alt+Enter inserts a newline for multi-line input."""
        event.current_buffer.insert_text('\n')

    # Ctrl+E to open in external editor
    @kb.add('c-e')
    def _(event):
        """Open the current input in an external editor (nano by default)."""
        event.current_buffer.open_in_editor()

    # Create history file path
    history_file = Path.home() / ".deepagents" / assistant_id / "history"
    history_file.parent.mkdir(parents=True, exist_ok=True)

    # Create the session
    session = PromptSession(
        message=HTML(f'<style fg="{COLORS["user"]}">></style> '),
        multiline=True,  # Keep multiline support but Enter submits
        history=FileHistory(str(history_file)),
        key_bindings=kb,
        completer=merge_completers([CommandCompleter(), BashCompleter(), FilePathCompleter()]),
        editing_mode=EditingMode.EMACS,
        complete_while_typing=True,  # Show completions as you type
        mouse_support=False,
        enable_open_in_editor=True,  # Allow Ctrl+X Ctrl+E to open external editor
    )

    return session


def render_todo_list(todos: list[dict]) -> None:
    """Render todo list as a rich Panel with checkboxes."""
    if not todos:
        return

    lines = []
    for todo in todos:
        status = todo.get("status", "pending")
        content = todo.get("content", "")

        if status == "completed":
            icon = "☑"
            style = "green"
        elif status == "in_progress":
            icon = "⏳"
            style = "yellow"
        else:  # pending
            icon = "☐"
            style = "dim"

        lines.append(f"[{style}]{icon} {content}[/{style}]")

    panel = Panel(
        "\n".join(lines),
        title="[bold]Task List[/bold]",
        border_style="cyan",
        box=box.ROUNDED,
        padding=(0, 1)
    )
    console.print(panel)


def prompt_for_shell_approval(action_request: dict) -> dict:
    """Prompt user to approve/reject/edit a shell command."""
    console.print()
    console.print(Panel(
        f"[bold yellow]⚠️  Shell Command Requires Approval[/bold yellow]\n\n"
        f"{action_request.get('description', 'No description available')}",
        border_style="yellow",
        box=box.ROUNDED,
        padding=(0, 1)
    ))
    console.print()

    while True:
        # Use input() instead of prompt() to avoid event loop conflicts
        choice = input("[A]pprove  [E]dit  [R]eject: ").strip().lower()

        if choice in ["a", "approve"]:
            return {"type": "approve"}
        elif choice in ["r", "reject"]:
            message = input("Reason for rejection (optional): ").strip()
            return {"type": "reject", "message": message or "User rejected the command"}
        elif choice in ["e", "edit"]:
            current_cmd = action_request['args'].get('command', '')
            console.print(f"Current command: [cyan]{current_cmd}[/cyan]")
            new_cmd = input("Edit command: ").strip()
            if not new_cmd:
                new_cmd = current_cmd  # Keep current if user enters nothing
            return {
                "type": "edit",
                "edited_action": {
                    "name": "shell",
                    "args": {"command": new_cmd}
                }
            }
        else:
            console.print("[yellow]Invalid choice. Please enter A, E, or R.[/yellow]")


def execute_task(user_input: str, agent, assistant_id: str | None, token_tracker: TokenTracker | None = None):
    """Execute any task by passing it directly to the AI agent."""
    console.print()

    # Parse file mentions and inject content if any
    prompt_text, mentioned_files = parse_file_mentions(user_input)

    if mentioned_files:
        context_parts = [prompt_text, "\n\n## Referenced Files\n"]
        for file_path in mentioned_files:
            try:
                content = file_path.read_text()
                # Limit file content to reasonable size
                if len(content) > 50000:
                    content = content[:50000] + "\n... (file truncated)"
                context_parts.append(f"\n### {file_path.name}\nPath: `{file_path}`\n```\n{content}\n```")
            except Exception as e:
                context_parts.append(f"\n### {file_path.name}\n[Error reading file: {e}]")

        final_input = "\n".join(context_parts)
    else:
        final_input = prompt_text

    config = {
        "configurable": {"thread_id": "main"},
        "metadata": {"assistant_id": assistant_id} if assistant_id else {}
    }

    has_responded = False
    current_text = ""
    printed_tool_calls_after_text = False
    captured_input_tokens = 0
    captured_output_tokens = 0
    current_todos = None  # Track current todo list state

    status = console.status(f"[bold {COLORS['thinking']}]Agent is thinking...", spinner="dots")
    status.start()
    spinner_active = True

    tool_icons = {
        "read_file": "📖",
        "write_file": "✏️",
        "edit_file": "✂️",
        "ls": "📁",
        "glob": "🔍",
        "grep": "🔎",
        "shell": "⚡",
        "web_search": "🌐",
        "http_request": "🌍",
        "task": "🤖",
        "write_todos": "📋",
    }

    # Stream input - may need to loop if there are interrupts
    stream_input = {"messages": [{"role": "user", "content": final_input}]}

    while True:
        interrupt_occurred = False
        hitl_response = None

        for chunk in agent.stream(
            stream_input,
            stream_mode="updates",  # Back to single mode like original
            subgraphs=True,
            config=config,
            durability="exit",
        ):
            # Unpack chunk - with subgraphs=True, it's (namespace, stream_mode, data)
            if isinstance(chunk, tuple) and len(chunk) == 3:
                _, _, data = chunk  # namespace and stream_mode not needed for updates-only
            elif isinstance(chunk, tuple) and len(chunk) == 2:
                # Fallback for non-subgraph mode
                _, data = chunk
            else:
                # Skip unexpected formats
                continue

            if not isinstance(data, dict):
                continue

            # Check for interrupts
            if "__interrupt__" in data:
                interrupt_data = data["__interrupt__"]
                if interrupt_data:
                    interrupt_obj = interrupt_data[0] if isinstance(interrupt_data, tuple) else interrupt_data
                    hitl_request = interrupt_obj.value if hasattr(interrupt_obj, 'value') else interrupt_obj

                    # Stop spinner for approval prompt
                    if spinner_active:
                        status.stop()
                        spinner_active = False

                    # Handle human-in-the-loop approval
                    decisions = []
                    for action_request in hitl_request.get("action_requests", []):
                        decision = prompt_for_shell_approval(action_request)
                        decisions.append(decision)

                    hitl_response = {"decisions": decisions}
                    interrupt_occurred = True
                    break

            # Extract chunk_data from updates
            chunk_data = list(data.values())[0] if data else None
            if not chunk_data:
                continue

            # Check for todo updates
            if "todos" in chunk_data:
                new_todos = chunk_data["todos"]
                if new_todos != current_todos:
                    current_todos = new_todos
                    # Stop spinner before rendering todos
                    if spinner_active:
                        status.stop()
                        spinner_active = False
                    console.print()
                    render_todo_list(new_todos)
                    console.print()

            # Check for messages in chunk_data
            if "messages" not in chunk_data:
                continue

            last_message = chunk_data["messages"][-1]
            message_role = getattr(last_message, "type", None)
            message_content = getattr(last_message, "content", None)

            # Skip tool results
            if message_role == "tool":
                continue

            # Handle AI messages
            if message_role == "ai":
                # Extract token usage if available
                if token_tracker:
                    usage = getattr(last_message, 'usage_metadata', None)
                    if not usage:
                        response_metadata = getattr(last_message, 'response_metadata', {})
                        usage = response_metadata.get('usage', None)

                    if usage:
                        input_toks = usage.get('input_tokens', 0)
                        output_toks = usage.get('output_tokens', 0)
                        if input_toks or output_toks:
                            captured_input_tokens = max(captured_input_tokens, input_toks)
                            captured_output_tokens = max(captured_output_tokens, output_toks)

                # First, extract and display text content
                text_content = ""

                if isinstance(message_content, str):
                    text_content = message_content
                elif isinstance(message_content, list):
                    for block in message_content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_content = block.get("text", "")
                            break

                if text_content.strip():
                    if spinner_active:
                        status.stop()
                        spinner_active = False

                    if not has_responded:
                        console.print("● ", style=COLORS["agent"], end="", markup=False)
                        has_responded = True
                        printed_tool_calls_after_text = False

                    if text_content != current_text:
                        new_text = text_content[len(current_text):]
                        console.print(new_text, style=COLORS["agent"], end="", markup=False)
                        current_text = text_content
                        printed_tool_calls_after_text = False

                # Then, handle tool calls from tool_calls attribute
                tool_calls = getattr(last_message, "tool_calls", None)
                if tool_calls:
                    # If we've printed text, ensure tool calls go on new line
                    if has_responded and current_text and not printed_tool_calls_after_text:
                        console.print()
                        printed_tool_calls_after_text = True

                    for tool_call in tool_calls:
                        tool_name = tool_call.get("name", "unknown")
                        tool_args = tool_call.get("args", {})

                        icon = tool_icons.get(tool_name, "🔧")
                        args_str = ", ".join(
                            f"{k}={truncate_value(str(v), 50)}" for k, v in tool_args.items()
                        )

                        if spinner_active:
                            status.stop()
                        console.print(f"  {icon} {tool_name}({args_str})", style=f"dim {COLORS['tool']}")
                        if spinner_active:
                            status.start()

                # Handle tool calls from content blocks (alternative format) - only if not already handled
                elif isinstance(message_content, list):
                    has_tool_use = False
                    for block in message_content:
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            has_tool_use = True
                            break

                    if has_tool_use:
                        # If we've printed text, ensure tool calls go on new line
                        if has_responded and current_text and not printed_tool_calls_after_text:
                            console.print()
                            printed_tool_calls_after_text = True

                        for block in message_content:
                            if isinstance(block, dict) and block.get("type") == "tool_use":
                                tool_name = block.get("name", "unknown")
                                tool_input = block.get("input", {})

                                icon = tool_icons.get(tool_name, "🔧")
                                args = ", ".join(
                                    f"{k}={truncate_value(str(v), 50)}" for k, v in tool_input.items()
                                )

                                if spinner_active:
                                    status.stop()
                                console.print(f"  {icon} {tool_name}({args})", style=f"dim {COLORS['tool']}")
                                if spinner_active:
                                    status.start()

        # After streaming loop - handle interrupt if it occurred
        if interrupt_occurred and hitl_response:
            # Resume the agent with the human decision
            stream_input = Command(resume=hitl_response)
            # Continue the while loop to restream
        else:
            # No interrupt, break out of while loop
            break

    if spinner_active:
        status.stop()

    if has_responded:
        console.print()

        # Display token usage if available
        if token_tracker and (captured_input_tokens or captured_output_tokens):
            token_tracker.add(captured_input_tokens, captured_output_tokens)
            token_tracker.display_last()

        console.print()


async def simple_cli(agent, assistant_id: str | None):
    """Main CLI loop."""
    console.clear()
    console.print(DEEP_AGENTS_ASCII, style=f"bold {COLORS['primary']}")
    console.print()

    if tavily_client is None:
        console.print(f"[yellow]⚠ Web search disabled:[/yellow] TAVILY_API_KEY not found.", style=COLORS["dim"])
        console.print(f"  To enable web search, set your Tavily API key:", style=COLORS["dim"])
        console.print(f"    export TAVILY_API_KEY=your_api_key_here", style=COLORS["dim"])
        console.print(f"  Or add it to your .env file. Get your key at: https://tavily.com", style=COLORS["dim"])
        console.print()

    console.print("... Ready to code! What would you like to build?", style=COLORS["agent"])
    console.print(f"  [dim]Working directory: {Path.cwd()}[/dim]")
    console.print()
    console.print(f"  Tips: Enter to submit, Alt+Enter for newline, Ctrl+E for editor, /help for commands", style=f"dim {COLORS['dim']}")
    console.print()

    # Create prompt session and token tracker
    session = create_prompt_session(assistant_id)
    token_tracker = TokenTracker()

    while True:
        try:
            user_input = await session.prompt_async()
            user_input = user_input.strip()
        except EOFError:
            break
        except KeyboardInterrupt:
            console.print()
            break

        if not user_input:
            continue

        # Check for slash commands first
        if user_input.startswith('/'):
            result = handle_command(user_input, agent, token_tracker)
            if result == 'exit':
                console.print(f"\nGoodbye!", style=COLORS["primary"])
                break
            elif result:
                # Command was handled, continue to next input
                continue

        # Check for bash commands (!)
        if user_input.startswith('!'):
            execute_bash_command(user_input)
            continue

        # Handle regular quit keywords
        if user_input.lower() in ["quit", "exit", "q"]:
            console.print(f"\nGoodbye!", style=COLORS["primary"])
            break

        execute_task(user_input, agent, assistant_id, token_tracker)


def list_agents():
    """List all available agents."""
    agents_dir = Path.home() / ".deepagents"
    
    if not agents_dir.exists() or not any(agents_dir.iterdir()):
        console.print("[yellow]No agents found.[/yellow]")
        console.print(f"[dim]Agents will be created in ~/.deepagents/ when you first use them.[/dim]", style=COLORS["dim"])
        return
    
    console.print(f"\n[bold]Available Agents:[/bold]\n", style=COLORS["primary"])
    
    for agent_path in sorted(agents_dir.iterdir()):
        if agent_path.is_dir():
            agent_name = agent_path.name
            agent_md = agent_path / "agent.md"
            
            if agent_md.exists():
                console.print(f"  • [bold]{agent_name}[/bold]", style=COLORS["primary"])
                console.print(f"    {agent_path}", style=COLORS["dim"])
            else:
                console.print(f"  • [bold]{agent_name}[/bold] [dim](incomplete)[/dim]", style=COLORS["tool"])
                console.print(f"    {agent_path}", style=COLORS["dim"])
    
    console.print()


def reset_agent(agent_name: str, source_agent: str = None):
    """Reset an agent to default or copy from another agent."""
    agents_dir = Path.home() / ".deepagents"
    agent_dir = agents_dir / agent_name
    
    if source_agent:
        source_dir = agents_dir / source_agent
        source_md = source_dir / "agent.md"
        
        if not source_md.exists():
            console.print(f"[bold red]Error:[/bold red] Source agent '{source_agent}' not found or has no agent.md")
            return
        
        source_content = source_md.read_text()
        action_desc = f"contents of agent '{source_agent}'"
    else:
        source_content = get_default_coding_instructions()
        action_desc = "default"
    
    if agent_dir.exists():
        shutil.rmtree(agent_dir)
        console.print(f"Removed existing agent directory: {agent_dir}", style=COLORS["tool"])
    
    agent_dir.mkdir(parents=True, exist_ok=True)
    agent_md = agent_dir / "agent.md"
    agent_md.write_text(source_content)
    
    console.print(f"✓ Agent '{agent_name}' reset to {action_desc}", style=COLORS["primary"])
    console.print(f"Location: {agent_dir}\n", style=COLORS["dim"])


def show_help():
    """Show help information."""
    console.print()
    console.print(DEEP_AGENTS_ASCII, style=f"bold {COLORS['primary']}")
    console.print()
    
    console.print("[bold]Usage:[/bold]", style=COLORS["primary"])
    console.print("  deepagents [--agent NAME]                      Start interactive session")
    console.print("  deepagents list                                List all available agents")
    console.print("  deepagents reset --agent AGENT                 Reset agent to default prompt")
    console.print("  deepagents reset --agent AGENT --target SOURCE Reset agent to copy of another agent")
    console.print("  deepagents help                                Show this help message")
    console.print()
    
    console.print("[bold]Examples:[/bold]", style=COLORS["primary"])
    console.print("  deepagents                              # Start with default agent", style=COLORS["dim"])
    console.print("  deepagents --agent mybot                # Start with agent named 'mybot'", style=COLORS["dim"])
    console.print("  deepagents list                         # List all agents", style=COLORS["dim"])
    console.print("  deepagents reset --agent mybot          # Reset mybot to default", style=COLORS["dim"])
    console.print("  deepagents reset --agent mybot --target other # Reset mybot to copy of 'other' agent", style=COLORS["dim"])
    console.print()
    
    console.print("[bold]Long-term Memory:[/bold]", style=COLORS["primary"])
    console.print("  By default, long-term memory is ENABLED using agent name 'agent'.", style=COLORS["dim"])
    console.print("  Memory includes:", style=COLORS["dim"])
    console.print("  - Persistent agent.md file with your instructions", style=COLORS["dim"])
    console.print("  - /memories/ folder for storing context across sessions", style=COLORS["dim"])
    console.print()
    
    console.print("[bold]Agent Storage:[/bold]", style=COLORS["primary"])
    console.print("  Agents are stored in: ~/.deepagents/AGENT_NAME/", style=COLORS["dim"])
    console.print("  Each agent has an agent.md file containing its prompt", style=COLORS["dim"])
    console.print()
    
    console.print("[bold]Interactive Features:[/bold]", style=COLORS["primary"])
    console.print("  Enter           Submit your message", style=COLORS["dim"])
    console.print("  Alt+Enter       Insert newline for multi-line (Option+Enter or ESC then Enter)", style=COLORS["dim"])
    console.print("  Ctrl+J          Insert newline (alternative)", style=COLORS["dim"])
    console.print("  Arrow keys      Navigate input and command history", style=COLORS["dim"])
    console.print("  @filename       Type @ to auto-complete files and inject content", style=COLORS["dim"])
    console.print("  /command        Type / to see available commands (auto-completes)", style=COLORS["dim"])
    console.print()

    console.print("[bold]Interactive Commands:[/bold]", style=COLORS["primary"])
    console.print("  /help           Show available commands and features", style=COLORS["dim"])
    console.print("  /clear          Clear screen and reset conversation", style=COLORS["dim"])
    console.print("  /tokens         Show token usage for current session", style=COLORS["dim"])
    console.print("  /quit, /exit    Exit the session", style=COLORS["dim"])
    console.print("  quit, exit, q   Exit the session (just type and press Enter)", style=COLORS["dim"])
    console.print()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="DeepAgents - AI Coding Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # List command
    subparsers.add_parser("list", help="List all available agents")
    
    # Help command
    subparsers.add_parser("help", help="Show help information")
    
    # Reset command
    reset_parser = subparsers.add_parser("reset", help="Reset an agent")
    reset_parser.add_argument("--agent", required=True, help="Name of agent to reset")
    reset_parser.add_argument("--target", dest="source_agent", help="Copy prompt from another agent")
    
    # Default interactive mode
    parser.add_argument(
        "--agent",
        default="agent",
        help="Agent identifier for separate memory stores (default: agent).",
    )
    
    return parser.parse_args()


async def main(assistant_id: str):
    """Main entry point."""

    # Create the model (checks API keys)
    model = create_model()

    # Create agent with conditional tools
    tools = [http_request]
    if tavily_client is not None:
        tools.append(web_search)

    shell_middleware = ShellToolMiddleware(
        workspace_root=os.getcwd(),
        execution_policy=HostExecutionPolicy()
    )

    # For long-term memory, point to ~/.deepagents/AGENT_NAME/ with /memories/ prefix
    agent_dir = Path.home() / ".deepagents" / assistant_id
    agent_dir.mkdir(parents=True, exist_ok=True)
    agent_md = agent_dir / "agent.md"
    if not agent_md.exists():
        source_content = get_default_coding_instructions()
        agent_md.write_text(source_content)

    # Long-term backend - rooted at agent directory
    # This handles both /memories/ files and /agent.md
    long_term_backend = FilesystemBackend(root_dir=agent_dir, virtual_mode=True)

    # Composite backend: current working directory for default, agent directory for /memories/
    backend = CompositeBackend(
        default=FilesystemBackend(),
        routes={"/memories/": long_term_backend}
    )

    # Use the same backend for agent memory middleware
    agent_middleware = [AgentMemoryMiddleware(backend=long_term_backend, memory_path="/memories/"), shell_middleware]
    system_prompt = f"""### Current Working Directory

The filesystem backend is currently operating in: `{Path.cwd()}`"""

    # Configure human-in-the-loop for shell commands
    shell_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject", "edit"],
        "description": lambda tool_call, state, runtime: (
            f"Shell Command: {tool_call['args'].get('command', 'N/A')}\n"
            f"Working Directory: {os.getcwd()}"
        )
    }

    agent = create_deep_agent(
        model=model,
        system_prompt=system_prompt,
        tools=tools,
        backend=backend,
        middleware=agent_middleware,
        interrupt_on={"shell": shell_interrupt_config},
    ).with_config(config)
    
    agent.checkpointer = InMemorySaver()
    
    try:
        await simple_cli(agent, assistant_id)
    except KeyboardInterrupt:
        console.print(f"\n\nGoodbye!", style=COLORS["primary"])
    except Exception as e:
        console.print(f"\n[bold red]❌ Error:[/bold red] {e}\n")


def cli_main():
    """Entry point for console script."""
    args = parse_args()

    if args.command == "help":
        show_help()
    elif args.command == "list":
        list_agents()
    elif args.command == "reset":
        reset_agent(args.agent, args.source_agent)
    else:
        # API key validation happens in create_model()
        asyncio.run(main(args.agent))


if __name__ == "__main__":
    cli_main()