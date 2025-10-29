"""DeepAgents Modal Sandbox Demo

Prerequisites:
- uv pip install deepagents modal openai
- modal setup
- OPENAI_API_KEY in .env
"""

import os
import sys
import logging

sys.path.insert(0, 'src')
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.WARNING, format='%(message)s')
logging.getLogger('deepagents.backends.sandbox.providers.modal_sandbox').setLevel(logging.INFO)

from deepagents import create_deep_agent
from deepagents.backends.sandbox import SandboxConfig


def print_tool_calls(result):
    for msg in result['messages']:
        # Show tool calls
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tc in msg.tool_calls:
                args_parts = []
                for k, v in tc['args'].items():
                    if isinstance(v, str) and len(v) > 60:
                        args_parts.append(f"{k}='{v[:60]}...'")
                    else:
                        args_parts.append(f"{k}={repr(v)}")
                args_str = ', '.join(args_parts)
                print(f"  🔧 {tc['name']}({args_str})")

        # Show tool results (especially bash output)
        if hasattr(msg, 'type') and msg.type == 'tool':
            content = msg.content
            if isinstance(content, str) and content.strip():
                # Truncate very long outputs
                if len(content) > 200:
                    content = content[:200] + '...'
                print(f"     → {content}")


def example_1_multi_turn():
    """Example 1: Multi-turn conversation"""
    print("\nExample 1: Multi-Turn Conversation\n")

    with create_deep_agent(
        sandbox=SandboxConfig(provider="modal"),
        system_prompt="You are a Python coding assistant. Use /workspace/ for code.",
        model="gpt-5-mini"
    ) as agent:
        msg1 = "Create /workspace/calculator.py with add and subtract functions"
        print(f"Turn 1: {msg1}")
        result1 = agent.invoke({"messages": [{"role": "user", "content": msg1}]})
        print_tool_calls(result1)

        msg2 = "Add multiply and divide to calculator.py"
        print(f"\nTurn 2: {msg2}")
        result2 = agent.invoke({"messages": [{"role": "user", "content": msg2}]})
        print_tool_calls(result2)


def example_2_modal_primary_with_memory():
    """Example 2: Modal primary + persistent /memory/"""
    print("\nExample 2: Modal Primary + Persistent /memory/")
    print("(Modal is default backend, /memory/ routed to a Store)\n")

    from deepagents.backends import CompositeBackend
    from deepagents.backends.store import StoreBackend
    from deepagents.backends.sandbox import create_sandbox_provider
    from langgraph.store.memory import InMemoryStore

    sandbox_config = SandboxConfig(provider="modal")
    memory_store = InMemoryStore()

    with create_sandbox_provider(sandbox_config) as modal_provider:
        def create_backend(runtime):
            return CompositeBackend(
                default=modal_provider,
                routes={"/memory/": StoreBackend(runtime)}
            )

        agent = create_deep_agent(
            backend=create_backend,
            store=memory_store,
            system_prompt="""You are a coding assistant.
- /memory/ → Persistent (file tools only)
- /workspace/ → Sandbox (file tools + bash)
Bash runs in container and cannot access /memory/.""",
            model="gpt-5-mini"
        )

        msg = "Save info to /memory/info.json with name='Claude' and surname='Code'. Create /workspace/demo.py that prints 'Hello!' and run it."
        print(f"User: {msg}")
        result = agent.invoke({"messages": [{"role": "user", "content": msg}]})
        print_tool_calls(result)

        # Show only the final agent response
        final_msg = result['messages'][-1]
        if hasattr(final_msg, 'content') and isinstance(final_msg.content, str):
            print(f"\n  Agent: {final_msg.content}")



def example_3_direct_provider():
    """Example 3: Direct provider access"""
    print("\nExample 3: Direct Provider Access\n")

    from deepagents.backends.sandbox import create_sandbox_provider

    with create_sandbox_provider(SandboxConfig(provider="modal")) as provider:
        provider.fs.write("/workspace/test.py", "print('Direct access!')")
        result = provider.execute("python /workspace/test.py", cwd="/workspace")
        print(f"Output: {result['stdout'].strip()}")


def main():
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: No API key found")
        sys.exit(1)

    # example_1_multi_turn()
    example_2_modal_primary_with_memory()
    # example_3_direct_provider()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
