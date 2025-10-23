#!/usr/bin/env python3
"""UNSTABLE CURRENTLY.
Beautiful Textual TUI for DeepAgents sandbox chat.

Usage:
    python chat_tui.py

Keyboard shortcuts:
    Ctrl+C - Quit
    Ctrl+L - Clear conversation
    Enter  - Send message (when in input field)
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from textual.app import App, ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Header, Footer, Static, Input
from textual.reactive import reactive
from rich.text import Text

from chat_agent import create_sandbox_chat_agent


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class Message:
    """Represents a single message in the conversation."""

    role: Literal["user", "agent", "tool_call", "tool_result"]
    content: str
    timestamp: float
    metadata: dict | None = None


# ============================================================================
# UTILITIES (from cli.py patterns)
# ============================================================================

MAX_ARG_LENGTH = 150
MAX_RESULT_LENGTH = 300

TOOL_ICONS = {
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


def truncate_value(value: str, max_length: int) -> str:
    """Truncate string with ellipsis if too long."""
    if len(value) > max_length:
        return value[:max_length] + "..."
    return value


def format_tool_args(tool_input: dict) -> str:
    """Format tool arguments for display."""
    if not tool_input:
        return ""

    args_parts = []
    for key, value in tool_input.items():
        value_str = truncate_value(str(value), MAX_ARG_LENGTH)
        args_parts.append(f"{key}={value_str}")

    return ", ".join(args_parts)


def _extract_text(content) -> str:
    """Normalize message content to a plain string."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n\n".join(part for part in parts if part).strip()
    return str(content).strip()


# ============================================================================
# TEXTUAL COMPONENTS
# ============================================================================


class WelcomeScreen(Static):
    """Splash screen with ASCII art."""

    def compose(self) -> ComposeResult:
        welcome_art = """[bold #10b981]
██████╗ ███████╗███████╗██████╗      █████╗  ██████╗ ███████╗███╗   ██╗████████╗███████╗
██╔══██╗██╔════╝██╔════╝██╔══██╗    ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝██╔════╝
██║  ██║█████╗  █████╗  ██████╔╝    ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   ███████╗
██║  ██║██╔══╝  ██╔══╝  ██╔══██╗    ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   ╚════██║
██████╔╝███████╗███████╗██║  ██║    ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   ███████║
╚═════╝ ╚══════╝╚══════╝╚═╝  ╚═╝    ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝

[white]Modal Sandboxes • GPT-5-mini[/white]
[dim]Press Enter to start[/dim]
"""
        yield Static(welcome_art, id="welcome")


class MessageBlock(Static):
    """Individual message in conversation."""

    def __init__(self, message: Message, **kwargs):
        super().__init__(**kwargs)
        self.message = message
        self.render_message()

    def render_message(self):
        """Render message with appropriate styling."""
        text = Text()
        role = self.message.role
        content = self.message.content.strip()

        if role == "user":
            text.append("> ", style="bold #10b981")
            text.append(content, style="bold white")
        elif role == "agent":
            text.append("• ", style="bold #34d399")
            text.append(content, style="white")
        elif role == "tool_call":
            text.append("↳ ", style="bold #a7f3d0")
            text.append(content, style="#a7f3d0")
        elif role == "tool_result":
            text.append("↳ ", style="bold #d1fae5")
            text.append(content, style="#d1fae5")
        else:
            text.append(content, style="white")

        self.update(text)

    def append_content(self, new_content: str):
        """Stream new content (for agent responses)."""
        self.message.content += new_content
        self.render_message()


class ConversationPanel(VerticalScroll):
    """Scrollable conversation container."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.can_focus = False
        self.messages: list[MessageBlock] = []

    async def add_message(self, message: Message) -> MessageBlock:
        """Add message and return block for streaming."""
        block = MessageBlock(message, classes="message")
        self.messages.append(block)
        await self.mount(block)
        self.scroll_end(animate=False)
        return block

    def clear_messages(self):
        """Clear all messages."""
        for msg in self.messages:
            msg.remove()
        self.messages.clear()


class WorkingIndicator(Static):
    """Animated working indicator with rich dots."""

    is_working = reactive(False)

    def watch_is_working(self, working: bool):
        """Update display based on working state."""
        if working:
            self.update("[#34d399]Agent is thinking…[/]")
        else:
            self.update("")


# ============================================================================
# MAIN APPLICATION
# ============================================================================


class SandboxChatApp(App):
    """Main TUI application."""

    CSS = """
    Screen {
        background: #050d0a;
        color: #e2f8ef;
    }

    #welcome {
        width: 100%;
        height: 100%;
        content-align: center middle;
    }

    #main-container {
        width: 100%;
        height: 100%;
    }

    #conversation {
        height: 1fr;
        padding: 1 2;
    }

    .message {
        margin: 0 0 1 0;
        padding: 0;
    }

    #indicator {
        height: auto;
        padding: 0 2;
        color: #6ee7b7;
    }

    Input {
        width: 100%;
        border: none;
        border-bottom: solid #10b981;
        background: #050d0a;
        color: #e2f8ef;
        padding: 0 0 1 0;
    }

    Input:focus {
        border: none;
        border-bottom: solid #34d399;
        outline: none;
        color: #34d399;
    }

    Header {
        background: #059669;
        color: #012e1d;
    }

    Footer {
        background: #050d0a;
        color: #3f7661;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+l", "clear", "Clear"),
    ]

    def __init__(self):
        super().__init__()
        self.showing_welcome = True
        self.agent = None
        self.config = {"configurable": {"thread_id": "sandbox-chat"}}
        self._welcome_widget: WelcomeScreen | None = None

    def compose(self) -> ComposeResult:
        """Initial composition."""
        yield Header(show_clock=True)
        self._welcome_widget = WelcomeScreen(id="welcome-screen")
        yield self._welcome_widget
        yield Footer()

    async def on_mount(self):
        """Setup on mount."""
        self.title = "DEEP AGENTS"
        self.sub_title = "Modal Sandboxes • GPT-5-mini"

        # Initialize agent in background
        self.agent = create_sandbox_chat_agent()
        asyncio.create_task(self._auto_start())

    async def on_key(self, event):
        """Handle welcome screen enter."""
        if self.showing_welcome and event.key == "enter":
            await self.start_conversation()

    async def _auto_start(self):
        """Start the conversation automatically after initial render."""
        await asyncio.sleep(0.1)
        if self.showing_welcome:
            await self.start_conversation()

    async def start_conversation(self):
        """Transition to chat interface."""
        if not self.showing_welcome:
            return
        self.showing_welcome = False

        # Remove welcome
        if self._welcome_widget is not None:
            await self._welcome_widget.remove()
            self._welcome_widget = None

        # Build chat UI
        container = Vertical(id="main-container")
        await self.mount(container)

        conversation = ConversationPanel(id="conversation")
        indicator = WorkingIndicator(id="indicator")
        input_widget = Input(placeholder="Type your message... (Enter to send)", id="input")

        await container.mount(conversation)
        await container.mount(indicator)
        await container.mount(input_widget)

        input_widget.focus()

        # Welcome message
        await conversation.add_message(
            Message(
                role="agent",
                content="👋 Ready to code! I can write Python scripts and execute them safely in Modal sandboxes. What would you like to build?",
                timestamp=datetime.now().timestamp(),
            )
        )

    async def on_input_submitted(self, event: Input.Submitted):
        """Handle message submission."""
        if not event.value.strip():
            return

        conversation = self.query_one("#conversation", ConversationPanel)
        indicator = self.query_one("#indicator", WorkingIndicator)
        input_widget = self.query_one("#input", Input)

        # Add user message
        user_msg = event.value
        await conversation.add_message(
            Message(
                role="user",
                content=user_msg,
                timestamp=datetime.now().timestamp(),
            )
        )

        # Clear input
        input_widget.value = ""

        # Show working indicator
        indicator.is_working = True

        # Process with agent
        await self.process_agent_response(conversation, user_msg)

        # Hide working indicator
        indicator.is_working = False

        input_widget.focus()

    async def process_agent_response(self, conversation: ConversationPanel, user_input: str):
        """Run the agent call in a background thread and display results."""

        def run_agent():
            return self.agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config=self.config,
            )

        result = await asyncio.to_thread(run_agent)
        messages = result.get("messages", [])

        for message in messages:
            message_role = getattr(message, "type", getattr(message, "role", ""))
            message_content = getattr(message, "content", "")

            if message_role in {"human", "user"}:
                continue

            if message_role == "ai":
                tool_calls = getattr(message, "tool_calls", None) or []
                for call in tool_calls:
                    tool_name = call.get("name", "tool")
                    tool_input = call.get("input", {})
                    icon = TOOL_ICONS.get(tool_name, "🔧")
                    args_str = format_tool_args(tool_input)
                    await conversation.add_message(
                        Message(
                            role="tool_call",
                            content=f"{icon} {tool_name}({args_str})",
                            timestamp=datetime.now().timestamp(),
                        )
                    )

                text_content = _extract_text(message_content)
                if text_content:
                    await conversation.add_message(
                        Message(
                            role="agent",
                            content=text_content,
                            timestamp=datetime.now().timestamp(),
                        )
                    )

            elif message_role == "tool":
                result_str = truncate_value(_extract_text(message_content), MAX_RESULT_LENGTH)
                await conversation.add_message(
                    Message(
                        role="tool_result",
                        content=result_str,
                        timestamp=datetime.now().timestamp(),
                    )
                )

    def action_clear(self):
        """Clear conversation."""
        try:
            conversation = self.query_one("#conversation", ConversationPanel)
            conversation.clear_messages()
        except Exception:
            pass

    def action_quit(self):
        """Quit application."""
        self.exit()


# ============================================================================
# ENTRY POINT
# ============================================================================


def main():
    """Run the TUI application."""
    app = SandboxChatApp()
    app.run()


if __name__ == "__main__":
    main()
