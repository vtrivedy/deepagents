"""Sandbox agent using OpenAI GPT-5 Mini with Modal sandboxes."""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent
from deepagents.middleware.sandbox_shell import SandboxShellMiddleware

# Load API keys from .env
load_dotenv()

# Explicit middleware instantiation (recommended for clarity)
sandbox_middleware = SandboxShellMiddleware(
    pip_packages=["pytest", "black", "ruff"],
    apt_packages=["git"],
    sync_cwd=False,  # Set to True to sync local files
    timeout=3600,  # 1 hour
    verbose=True,
)

# Create the agent with OpenAI and Modal sandbox
agent = create_deep_agent(
    model=ChatOpenAI(
        model="gpt-5-mini",
        temperature=0.7,
    ),
    middleware=[sandbox_middleware],
    system_prompt="""You are a coding assistant that can safely execute shell commands in an isolated sandbox.

The sandbox is a secure Modal container with:
- Python 3.11
- pytest, black, ruff (pre-installed)
- git (pre-installed)
- Working directory: /workspace

Use the shell tool to run commands safely without affecting the user's local system.""",
)

# Alternative: Use convenience parameter for simpler setup
# agent = create_deep_agent(
#     model=ChatOpenAI(model="gpt-4o-mini-2025-08-07", temperature=0.7),
#     use_sandbox_shell=True,
#     sandbox_config={
#         "pip_packages": ["pytest", "black", "ruff"],
#         "apt_packages": ["git"],
#         "sync_cwd": False,
#         "timeout": 3600,
#         "verbose": True,
#     },
#     system_prompt="...",
# )
