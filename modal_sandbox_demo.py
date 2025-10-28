"""
DeepAgents Modal Sandbox Demo Script

ARCHITECTURE:
-------------
SandboxConfig(provider="modal")
    ↓
create_deep_agent()
    ↓
creates ModalSandboxProvider (from deepagents.backends.sandbox.providers.modal_sandbox)
    ↓
wraps in SandboxBackend (from deepagents.backends.sandbox.backend)
    ↓
wraps in CompositeBackend (StateBackend for /, SandboxBackend for /workspace/)
    ↓
agent.invoke() → ModalSandboxProvider.ensure_ready() → starts ephemeral Modal app + sandbox
    ↓
agent.cleanup() → ModalSandboxProvider.cleanup() → terminates sandbox + stops app

Prerequisites:
1. pip install deepagents modal openai  # or anthropic
2. modal setup
3. Create .env file with: OPENAI_API_KEY=sk-...  # or ANTHROPIC_API_KEY
"""

import os
import sys
import logging

# Add src to path so we can import deepagents
sys.path.insert(0, 'src')

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging - only show important Modal sandbox logs
logging.basicConfig(level=logging.WARNING, format='%(message)s')
# Enable INFO for our sandbox provider only (hide httpx noise)
logging.getLogger('deepagents.backends.sandbox.providers.modal_sandbox').setLevel(logging.INFO)

from deepagents import create_deep_agent
from deepagents.backends.sandbox import SandboxConfig
from langchain_core.tools import tool


def example_1_basic():
    """Example 1: Basic sandbox usage - create a simple file"""
    print("\n=== Example 1: Basic sandbox usage ===\n")

    agent = create_deep_agent(
        sandbox=SandboxConfig(provider="modal"),
        system_prompt="You are a Python coding assistant. Use /workspace/ for code.",
        model="gpt-5-mini"
    )

    user_msg = "Create a hello.py file in /workspace/ that prints 'Hello from Modal sandbox!'"
    print(f"User: {user_msg}\n")

    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": user_msg
        }]
    })

    print(f"\nAgent: {result['messages'][-1].content}\n")
    agent.cleanup()


def example_2_multi_turn():
    """Example 2: Multi-turn conversation - same sandbox persists"""
    print("\n=== Example 2: Multi-turn conversation (same sandbox persists) ===\n")

    agent = create_deep_agent(
        sandbox=SandboxConfig(provider="modal"),
        system_prompt="You are a Python coding assistant. Use /workspace/ for code.",
        model="gpt-5-mini"
    )

    msg1 = "Create calculator.py in /workspace/ with add and subtract functions"
    print(f"User (Turn 1): {msg1}\n")

    result1 = agent.invoke({
        "messages": [{"role": "user", "content": msg1}]
    })
    print(f"\nAgent (Turn 1): {result1['messages'][-1].content}\n")

    msg2 = "Add multiply and divide functions to calculator.py"
    print(f"\nUser (Turn 2): {msg2}\n")

    result2 = agent.invoke({
        "messages": [{"role": "user", "content": msg2}]
    })
    print(f"\nAgent (Turn 2): {result2['messages'][-1].content}\n")
    agent.cleanup()

def example_3_multi_filesystem():
    """Example 3: Multi-filesystem support + automatic bash execution"""
    print("\n=== Example 3: Multi-filesystem + Bash (Auto-included) ===\n")

    from deepagents.backends import CompositeBackend, StateBackend
    from deepagents.backends.sandbox import SandboxBackend
    from deepagents.backends.sandbox.providers.modal_sandbox import ModalSandboxProvider

    # 1. Create sandbox backend for /workspace/
    sandbox_config = SandboxConfig(provider="modal")
    modal_provider = ModalSandboxProvider.from_config(sandbox_config)
    sandbox_backend = SandboxBackend(modal_provider)

    # 2. Create CompositeBackend factory that routes paths to different backends
    def create_composite_backend(runtime):
        return CompositeBackend(
            default=StateBackend(runtime),      # / → local state (in-memory)
            routes={
                "/workspace/": sandbox_backend  # /workspace/ → Modal sandbox
            }
        )

    # 3. Create bash tool for the sandbox (when using explicit backend, need to add manually)
    @tool
    def bash(command: str) -> str:
        """Execute a bash command in the Modal sandbox. Use this to run Python scripts or shell commands.

        Args:
            command: The bash command to execute (e.g., 'python /workspace/script.py')
        """
        import asyncio
        result = asyncio.run(modal_provider.execute(command, cwd="/workspace"))

        if result["exit_code"] != 0:
            return f"Error (exit {result['exit_code']}):\n{result['stderr']}"
        return result["stdout"]

    # 4. Create agent with explicit backend + bash tool
    # Note: When using explicit backend (not sandbox=), bash tool must be added manually
    agent = create_deep_agent(
        backend=create_composite_backend,
        tools=[bash],  # Must add bash manually when using explicit backend
        system_prompt="""You are a coding assistant.
- /workspace/ = Modal sandbox (remote code execution)
- / = local state (in-memory notes)
- Use the bash tool to execute Python scripts in /workspace/""",
        model="gpt-4o-mini"
    )

    user_msg = "Create ASCII art: 1) plan in /plan.txt, 2) code in /workspace/ascii.py, 3) RUN it with bash tool"
    print(f"User: {user_msg}\n")

    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": user_msg
        }]
    })

    print(f"\nAgent: {result['messages'][-1].content}\n")

    # Cleanup sandbox
    sandbox_backend.cleanup()


def example_4_direct_fs_access():
    """Example 4: Direct filesystem access without an agent"""
    print("\n=== Example 4: Direct filesystem access (no agent required) ===\n")

    import asyncio
    from deepagents.backends.sandbox.providers.modal_sandbox import ModalSandboxProvider

    config = SandboxConfig(provider="modal")
    provider = ModalSandboxProvider.from_config(config)

    # Get the filesystem interface - provides clean, sync API
    fs = provider.fs

    print("Writing /workspace/direct.py...\n")
    fs.write("/workspace/direct.py", "print('Direct upload!')\n")

    print("Reading /workspace/direct.py...")
    content = fs.read_text("/workspace/direct.py")
    print(f"Content: {content.strip()}\n")

    print("Listing /workspace/...")
    files = fs.list("/workspace/")
    for f in files:
        print(f"  {f['path']} ({f['size']}b)")
    print()

    # Cleanup sandbox
    asyncio.run(provider.cleanup())


def example_5_interactive_chat():
    """Example 5: Interactive chat session"""
    print("\n=== Example 5: Interactive chat session ===")
    print("Type 'exit', 'quit', or 'bye' to end.\n")

    agent = create_deep_agent(
        sandbox=SandboxConfig(provider="modal"),
        system_prompt="You are a helpful Python coding assistant. /workspace/ for code, / for notes. Be concise!",
        model="gpt-5-mini"
    )

    messages = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.lower() in ['exit', 'quit', 'bye', 'q'] or not user_input:
            break

        messages.append({"role": "user", "content": user_input})
        result = agent.invoke({"messages": messages})

        assistant_message = result["messages"][-1].content
        print(f"\nAgent: {assistant_message}")

        messages.append({"role": "assistant", "content": assistant_message})

    print("\nEnding session...\n")
    agent.cleanup()


def main():
    """Uncomment the examples you want to run"""
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: No API key found! Set OPENAI_API_KEY or ANTHROPIC_API_KEY")
        sys.exit(1)

    # Uncomment examples to run:
    # example_1_basic()
    # example_2_multi_turn()
    # example_3_multi_filesystem()
    example_4_direct_fs_access()
    # example_5_interactive_chat()  # Interactive loop - run last


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
