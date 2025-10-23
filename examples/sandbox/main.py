#!/usr/bin/env python3
"""Minimal test script for Modal sandbox integration with OpenAI."""

from sandbox_agent import agent


def _extract_text(content):
    """Normalize message content to a printable string."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        return "\n".join(filter(None, parts)).strip()
    return str(content).strip()


def main():
    """Run a simple test of the sandbox agent."""
    print("🚀 Testing Modal Sandbox with OpenAI GPT-5 Mini")
    print("=" * 60)

    # Simple test message
    user_message = "Run 'pytest --version' to verify pytest is installed in the sandbox.  explain exactly how you know!"

    print(f"\n📤 User: {user_message}\n")

    # Invoke agent
    result = agent.invoke({"messages": [{"role": "user", "content": user_message}]})

    # Show full tool + message trace
    print("🧾 Conversation Trace:")
    print("-" * 60)
    for message in result["messages"]:
        role = getattr(message, "type", getattr(message, "role", ""))

        if role == "human":
            print(f"👤 User:\n{_extract_text(message.content)}\n")
            continue

        if role == "ai":
            tool_calls = getattr(message, "tool_calls", None) or []
            if tool_calls:
                print("🤖 Assistant issued tool call(s):")
                for call in tool_calls:
                    name = call.get("name", "unknown_tool")
                    args = call.get("args") or call.get("arguments") or {}
                    print(f"   • {name}({args})")
                text = _extract_text(message.content)
                if text:
                    print(f"\n🤖 Assistant note(s):\n{text}\n")
            else:
                print(f"🤖 Assistant:\n{_extract_text(message.content)}\n")
            continue

        if role == "tool":
            tool_name = getattr(message, "name", "tool")
            print(f"🔧 Tool Result [{tool_name}]:\n{_extract_text(message.content)}\n")
            continue

        # Fallback for any other message types
        print(f"• {role or 'message'}:\n{_extract_text(getattr(message, 'content', ''))}\n")

    # Print final assistant response
    print("🤖 Final Assistant Response:")
    print("-" * 60)
    print(_extract_text(result["messages"][-1].content))
    print("-" * 60)

    print("\n✅ Test complete!")


if __name__ == "__main__":
    main()
