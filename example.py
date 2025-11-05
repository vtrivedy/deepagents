"""Minimal example showing how to create a DeepAgent.

This file demonstrates creating a simple deep agent using create_deep_agent
and storing a minimal serialized representation to disk (example_agent.json).

Run as:
    python example.py

Note: the full agent object may not be JSON-serializable. This example attempts
to call `to_dict()` if available, otherwise it stores a minimal metadata dict.
"""

from deepagents import create_deep_agent
import json


def create_example_agent():
    """Create a minimal deep agent with a tiny system prompt.

    Returns:
        The created agent object.
    """
    agent = create_deep_agent(
        system_prompt="You are a helpful assistant for demo/example purposes.",
        name="example-deep-agent",
    )
    return agent


if __name__ == "__main__":
    agent = create_example_agent()

    # Attempt to serialize a minimal representation of the agent. The full
    # agent object might not be JSON-serializable or picklable depending on
    # backends and model objects, so we prefer a lightweight fallback.
    out = None
    try:
        if hasattr(agent, "to_dict"):
            out = agent.to_dict()
        else:
            # Fallback: store common metadata if present
            out = {
                "name": getattr(agent, "name", None),
                "repr": repr(agent),
            }
    except Exception as e:
        out = {"error": "failed to serialize agent", "exception": str(e)}

    with open("example_agent.json", "w") as f:
        json.dump(out, f, indent=2)

    print("Wrote example_agent.json (minimal agent representation).")
