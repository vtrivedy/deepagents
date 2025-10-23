"""Agent configuration for TUI sandbox chat.

This module provides the agent setup used by the Textual TUI.
Kept separate for modularity and easy configuration changes.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent
from deepagents.middleware.sandbox_shell import SandboxShellMiddleware
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

# Get path to the local file we want to add to sandbox
CURRENT_DIR = Path(__file__).parent
LOCAL_FILE = CURRENT_DIR / "possibly_unsafe_code.py"


def create_sandbox_chat_agent():
    """Create agent with Modal sandbox for code execution.

    Returns:
        Configured agent ready for TUI interaction.
    """
    # Configure sandbox for Python development
    sandbox_middleware = SandboxShellMiddleware(
        pip_packages=["pytest", "requests", "numpy", "pandas"],
        apt_packages=["git"],
        sync_paths=[str(LOCAL_FILE)],  # Add local file to sandbox
        timeout=3600,
        verbose=True,
    )

    # Pre-create the sandbox now so verbose output appears during initialization
    # This prevents sandbox creation messages from appearing mid-chat
    if sandbox_middleware.sb is None:
        sandbox_middleware.sb = sandbox_middleware._create_sandbox()

    # Create agent with coding focus
    agent = create_deep_agent(
        model=ChatOpenAI(
            model="gpt-5-mini",
            temperature=0.7,
        ),
        middleware=[sandbox_middleware],
        system_prompt="""You are a Python coding assistant with Modal sandbox access.

The sandbox environment:
- Python 3.11
- pytest, requests, numpy, pandas (pre-installed)
- git (pre-installed)
- Working directory: /workspace
- Pre-loaded file: possibly_unsafe_code.py (available at /workspace/possibly_unsafe_code.py)

You can write scripts, execute them safely, and show results.
Keep responses concise and focused.""",
    ).with_config({"recursion_limit": 1000})

    # Add checkpointer for conversation state
    agent.checkpointer = InMemorySaver()

    return agent
