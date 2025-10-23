#!/usr/bin/env python3
"""Minimalist prompt-toolkit TUI for DeepAgents sandbox chat.

Beautiful emerald green theme with streaming responses.

Controls:
- Enter: Submit message
- Alt-Enter (or Esc then Enter): New line for multiline input
- Ctrl+C or Ctrl+D: Quit
"""

import asyncio

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.spinner import Spinner

from chat_agent import create_sandbox_chat_agent


# ============================================================================
# CONFIGURATION
# ============================================================================

COLORS = {
    "primary": "#10b981",    # Emerald 500
    "dim": "#6b7280",        # Gray 500
    "user": "#ffffff",       # White
    "agent": "#10b981",      # Emerald 500
    "thinking": "#34d399",   # Emerald 400
    "tool": "#fbbf24",       # Amber 400
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

# ============================================================================
# GLOBALS
# ============================================================================

console = Console()
agent = None
config = {"configurable": {"thread_id": "sandbox-chat"}}

# ============================================================================
# UTILITIES
# ============================================================================

MAX_ARG_LENGTH = 150

TOOL_ICONS = {
    "shell": "⚡",
    "write_file": "✏️",
    "read_file": "📖",
    "edit_file": "✂️",
    "ls": "📁",
    "glob": "🔍",
    "grep": "🔎",
}


def truncate(text: str, max_len: int) -> str:
    """Truncate text with ellipsis."""
    return text[:max_len] + "..." if len(text) > max_len else text


# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================


def show_welcome():
    """Display welcome screen and wait for Enter."""
    console.clear()
    console.print(DEEP_AGENTS_ASCII, style=f"bold {COLORS['primary']}")
    console.print("\n")
    console.print("Press Enter to start", style=COLORS["dim"])
    input()
    # Don't clear - keep ASCII art visible!


# ============================================================================
# AGENT INTERACTION
# ============================================================================


async def stream_agent_response(user_input: str):
    """Stream agent response using async iteration."""
    global agent

    has_responded = False
    current_text = ""

    # Start spinner manually so we can stop it when we have content
    status = console.status("[bold #34d399]Agent is thinking...", spinner="dots")
    status.start()
    spinner_active = True

    async for _, chunk in agent.astream(
        {"messages": [{"role": "user", "content": user_input}]},
        stream_mode="updates",
        subgraphs=True,
        config=config,
        durability="exit",
    ):
        chunk_data = list(chunk.values())[0]
        if not chunk_data or "messages" not in chunk_data:
            continue

        last_message = chunk_data["messages"][-1]
        message_role = getattr(last_message, "type", None)
        message_content = getattr(last_message, "content", None)

        # Handle tool calls from AI messages (LangChain tool_calls attribute)
        tool_calls = getattr(last_message, "tool_calls", None)
        if tool_calls and message_role == "ai":
            for tool_call in tool_calls:
                tool_name = tool_call.get("name", "unknown")
                tool_args = tool_call.get("args", {})

                icon = TOOL_ICONS.get(tool_name, "🔧")
                args_str = ", ".join(
                    f"{k}={truncate(str(v), 50)}" for k, v in tool_args.items()
                )

                # Stop spinner temporarily to print tool call
                if spinner_active:
                    status.stop()
                console.print(f"  {icon} {tool_name}({args_str})", style=f"dim {COLORS['tool']}")
                # Restart spinner for next tool/processing
                if spinner_active:
                    status.start()

        # Skip tool results - they're verbose and the agent will summarize
        if message_role == "tool":
            continue

        if not message_content:
            continue

        # Handle tool calls from content blocks (alternative format)
        if message_role == "ai" and isinstance(message_content, list):
            for block in message_content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_name = block.get("name", "unknown")
                    tool_input = block.get("input", {})

                    icon = TOOL_ICONS.get(tool_name, "🔧")
                    args = ", ".join(
                        f"{k}={truncate(str(v), 50)}" for k, v in tool_input.items()
                    )

                    # Stop spinner temporarily to print tool call
                    if spinner_active:
                        status.stop()
                    console.print(f"  {icon} {tool_name}({args})", style=f"dim {COLORS['tool']}")
                    # Restart spinner for next tool/processing
                    if spinner_active:
                        status.start()

        # Handle agent text responses
        if message_role == "ai":
            text_content = ""

            if isinstance(message_content, str):
                text_content = message_content
            elif isinstance(message_content, list):
                for block in message_content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_content = block.get("text", "")
                        break

            if text_content.strip():
                # Stop spinner when we have actual text to display
                if spinner_active:
                    status.stop()
                    spinner_active = False

                # Print prefix on first response
                if not has_responded:
                    console.print("... ", style=COLORS["agent"], end="")
                    has_responded = True

                # Stream new content
                if text_content != current_text:
                    new_text = text_content[len(current_text) :]
                    console.print(new_text, style=COLORS["agent"], end="")
                    current_text = text_content

    # Make sure spinner is stopped (in case loop ended without content)
    if spinner_active:
        status.stop()

    if has_responded:
        console.print()  # Newline
        console.print()  # Blank line


# ============================================================================
# MAIN
# ============================================================================


async def main():
    """Main entry point."""
    global agent

    # Show welcome
    show_welcome()

    # Initialize agent
    console.print("\nInitializing agent...", style=COLORS["dim"])
    agent = create_sandbox_chat_agent()

    # Show ready message
    console.print("\n... Ready to code! What would you like to build?", style=COLORS["agent"])
    console.print()

    # One-time hint for multiline input
    console.print("  Tip: Alt-Enter for newline, Enter to submit", style=f"dim {COLORS['dim']}")

    # Setup key bindings for multiline input
    kb = KeyBindings()

    @kb.add('enter')
    def _(event):
        buffer = event.current_buffer
        if buffer.text.strip():  # Only submit if buffer has content
            buffer.validate_and_handle()
        # else: do nothing - no newline, no submission

    @kb.add('escape', 'enter')  # Alt-Enter (or Esc then Enter) for newline
    def _(event):
        event.current_buffer.insert_text('\n')

    # Setup prompt session with multiline support
    style = Style.from_dict({"prompt": COLORS["user"]})
    session = PromptSession(
        message="> ",
        style=style,
        multiline=True,
        prompt_continuation=lambda width, line_number, is_soft_wrap: "  ",
        key_bindings=kb,
    )

    # Main chat loop
    while True:
        try:
            # Get user input
            user_input = await session.prompt_async()

            # Handle quit commands
            if user_input.strip().lower() in ["quit", "exit", "q"]:
                break

            # Skip empty input
            if not user_input.strip():
                continue

            # Add spacing
            console.print()

            # Stream agent response
            await stream_agent_response(user_input)

        except (KeyboardInterrupt, EOFError):
            break

    # Goodbye message
    console.print("\nGoodbye!", style=COLORS["primary"])


if __name__ == "__main__":
    asyncio.run(main())
