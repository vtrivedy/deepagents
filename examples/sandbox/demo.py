#!/usr/bin/env python3
"""Demo of SandboxShellMiddleware with various configurations.

This example shows different ways to use Modal sandboxes with DeepAgents:
1. Basic sandbox with pip packages
2. File sync from local directory
3. Custom Modal image
"""

import os

from deepagents import create_deep_agent


def example_1_basic():
    """Example 1: Basic sandbox with pip packages."""
    print("=" * 60)
    print("Example 1: Basic Sandbox with Packages")
    print("=" * 60)

    agent = create_deep_agent(
        use_sandbox_shell=True,
        sandbox_config={
            "pip_packages": ["pytest", "black"],
            "verbose": True,
        },
    )

    result = agent.invoke({"messages": [{"role": "user", "content": "Run pytest --version to verify installation"}]})

    print("\nAgent response:")
    print(result["messages"][-1].content)
    print()


def example_2_file_sync():
    """Example 2: Sync local files to sandbox."""
    print("=" * 60)
    print("Example 2: File Sync")
    print("=" * 60)

    # Create a test file to sync
    test_file = "test_hello.py"
    with open(test_file, "w") as f:
        f.write("print('Hello from sandbox!')\n")

    try:
        agent = create_deep_agent(
            use_sandbox_shell=True,
            sandbox_config={
                "sync_cwd": True,
                "sync_exclude": ["*.pyc", "__pycache__", ".git", "venv", "node_modules"],
                "pip_packages": ["pytest"],
                "verbose": True,
            },
        )

        result = agent.invoke(
            {"messages": [{"role": "user", "content": "List files in /workspace and then run the test_hello.py file"}]}
        )

        print("\nAgent response:")
        print(result["messages"][-1].content)
        print()

    finally:
        # Cleanup test file
        if os.path.exists(test_file):
            os.remove(test_file)


def example_3_custom_image():
    """Example 3: Custom Modal image."""
    print("=" * 60)
    print("Example 3: Custom Image")
    print("=" * 60)

    try:
        import modal

        custom_img = modal.Image.debian_slim().pip_install("requests", "beautifulsoup4")

        agent = create_deep_agent(
            use_sandbox_shell=True,
            sandbox_config={
                "custom_image": custom_img,
                "verbose": True,
            },
        )

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Verify that requests and beautifulsoup4 are installed by importing them",
                    }
                ]
            }
        )

        print("\nAgent response:")
        print(result["messages"][-1].content)
        print()

    except ImportError:
        print("⚠️  modal package not installed. Run: pip install modal")


def example_4_interactive():
    """Example 4: Interactive session with persistent sandbox."""
    print("=" * 60)
    print("Example 4: Interactive Session")
    print("=" * 60)

    agent = create_deep_agent(
        use_sandbox_shell=True,
        sandbox_config={
            "pip_packages": ["pytest"],
            "sync_cwd": True,
            "verbose": True,
        },
    )

    # Multiple commands in same sandbox session
    commands = [
        "Create a simple Python script that calculates fibonacci numbers",
        "Run the script you just created",
        "Install numpy and use it to calculate the same fibonacci numbers",
    ]

    for i, cmd in enumerate(commands, 1):
        print(f"\n--- Command {i}: {cmd} ---")
        result = agent.invoke({"messages": [{"role": "user", "content": cmd}]})
        print(result["messages"][-1].content)


def main():
    """Run all examples."""
    print("\n🚀 DeepAgents Sandbox Shell Examples\n")

    try:
        # Run examples
        example_1_basic()
        input("\nPress Enter to continue to Example 2...")

        example_2_file_sync()
        input("\nPress Enter to continue to Example 3...")

        example_3_custom_image()
        input("\nPress Enter to continue to Example 4...")

        example_4_interactive()

    except KeyboardInterrupt:
        print("\n\n⚠️  Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
    finally:
        print("\n✅ Examples completed")


if __name__ == "__main__":
    main()
