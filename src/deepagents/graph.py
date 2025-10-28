"""Deepagents come with planning, filesystem, and subagents."""

from collections.abc import Callable, Sequence
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, InterruptOnConfig, TodoListMiddleware
from langchain.agents.middleware.summarization import SummarizationMiddleware
from langchain.agents.middleware.types import AgentMiddleware
from langchain.agents.structured_output import ResponseFormat
from langchain_anthropic import ChatAnthropic
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, tool
from langgraph.cache.base import BaseCache
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer

from deepagents.backends.protocol import BackendProtocol, BackendFactory
from deepagents.backends import CompositeBackend, StateBackend
from deepagents.backends.sandbox import SandboxBackend, SandboxConfig, create_sandbox_provider
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from deepagents.middleware.subagents import CompiledSubAgent, SubAgent, SubAgentMiddleware

BASE_AGENT_PROMPT = "In order to complete the objective that the user asks of you, you have access to a number of standard tools."


class DeepAgent:
    """Wrapper for deep agent that manages sandbox lifecycle.

    This wraps a LangGraph compiled graph and handles sandbox cleanup.

    Example:
        agent = create_deep_agent(sandbox=SandboxConfig(provider="modal"))
        agent.invoke({"messages": [...]})
        agent.cleanup()

        # Or use context manager
        with create_deep_agent(sandbox=...) as agent:
            agent.invoke(...)
        # Cleanup automatic
    """

    def __init__(self, graph: CompiledStateGraph, sandbox: SandboxBackend | None = None):
        self.graph = graph
        self._sandbox = sandbox

    def invoke(self, *args, **kwargs):
        """Invoke the agent graph"""
        return self.graph.invoke(*args, **kwargs)

    def stream(self, *args, **kwargs):
        """Stream the agent graph"""
        return self.graph.stream(*args, **kwargs)

    def ainvoke(self, *args, **kwargs):
        """Async invoke the agent graph"""
        return self.graph.ainvoke(*args, **kwargs)

    def astream(self, *args, **kwargs):
        """Async stream the agent graph"""
        return self.graph.astream(*args, **kwargs)

    def cleanup(self):
        """Clean up sandbox resources"""
        if self._sandbox:
            self._sandbox.cleanup()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup sandbox"""
        self.cleanup()
        return False


def get_default_model() -> ChatAnthropic:
    """Get the default model for deep agents.

    Returns:
        ChatAnthropic instance configured with Claude Sonnet 4.
    """
    return ChatAnthropic(
        model_name="claude-sonnet-4-5-20250929",
        max_tokens=20000,
    )


def create_deep_agent(
    model: str | BaseChatModel | None = None,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    *,
    system_prompt: str | None = None,
    middleware: Sequence[AgentMiddleware] = (),
    subagents: list[SubAgent | CompiledSubAgent] | None = None,
    response_format: ResponseFormat | None = None,
    context_schema: type[Any] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    sandbox: SandboxConfig | None = None,
    backend: BackendProtocol | BackendFactory | None = None,
    interrupt_on: dict[str, bool | InterruptOnConfig] | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache | None = None,
) -> DeepAgent | CompiledStateGraph:
    """Create a deep agent.

    This agent will by default have access to a tool to write todos (write_todos),
    six file editing tools: write_file, ls, read_file, edit_file, glob_search, grep_search,
    and a tool to call subagents.

    Args:
        model: The model to use. Defaults to Claude Sonnet 4.
        tools: The tools the agent should have access to.
        system_prompt: The additional instructions the agent should have. Will go in
            the system prompt.
        middleware: Additional middleware to apply after standard middleware.
        subagents: The subagents to use. Each subagent should be a dictionary with the
            following keys:
                - `name`
                - `description` (used by the main agent to decide whether to call the
                  sub agent)
                - `prompt` (used as the system prompt in the subagent)
                - (optional) `tools`
                - (optional) `model` (either a LanguageModelLike instance or dict
                  settings)
                - (optional) `middleware` (list of AgentMiddleware)
        response_format: A structured output response format to use for the agent.
        context_schema: The schema of the deep agent.
        checkpointer: Optional checkpointer for persisting agent state between runs.
        store: Optional store for persistent storage (required if backend uses StoreBackend).
        sandbox: Optional sandbox config for isolated code execution. If provided, automatically
            creates a sandbox backend with local + sandbox filesystem routing.
            Agent will manage sandbox lifecycle. Call agent.cleanup() when done or use
            context manager: `with create_deep_agent(sandbox=...) as agent:`
        backend: Optional backend for file storage. Pass either a Backend instance or a
            callable factory like `lambda rt: StateBackend(rt)`. Cannot be used with sandbox.
        interrupt_on: Optional Dict[str, bool | InterruptOnConfig] mapping tool names to
            interrupt configs.
        debug: Whether to enable debug mode. Passed through to create_agent.
        name: The name of the agent. Passed through to create_agent.
        cache: The cache to use for the agent. Passed through to create_agent.

    Returns:
        A DeepAgent wrapper (if sandbox provided) or CompiledStateGraph (if backend provided).
    """
    if sandbox is not None and backend is not None:
        raise ValueError("Cannot specify both 'sandbox' and 'backend' parameters")

    # Create sandbox backend if config provided
    sandbox_backend = None
    if sandbox is not None:
        provider = create_sandbox_provider(sandbox)
        sandbox_backend = SandboxBackend(provider)
        backend = lambda rt: CompositeBackend(
            default=StateBackend(rt),
            routes={"/workspace/": sandbox_backend}
        )

        # Auto-add bash tool for sandbox execution
        @tool
        def bash(command: str) -> str:
            """Execute a bash command in the sandbox container. Use this to run Python scripts, shell commands, or any executable code.

            Args:
                command: The bash command to execute (e.g., 'python /workspace/script.py', 'ls -la', 'cat file.txt')

            Returns:
                The stdout from the command execution, or an error message if the command failed.
            """
            import asyncio
            result = asyncio.run(provider.execute(command, cwd="/workspace"))

            if result["exit_code"] != 0:
                return f"Command failed with exit code {result['exit_code']}:\nstderr: {result['stderr']}\nstdout: {result['stdout']}"
            return result["stdout"]

        # Add bash tool to the tools list
        tools = list(tools) if tools else []
        tools.append(bash)
    if model is None:
        model = get_default_model()

    deepagent_middleware = [
        TodoListMiddleware(),
        FilesystemMiddleware(backend=backend),
        SubAgentMiddleware(
            default_model=model,
            default_tools=tools,
            subagents=subagents if subagents is not None else [],
            default_middleware=[
                TodoListMiddleware(),
                FilesystemMiddleware(backend=backend),
                SummarizationMiddleware(
                    model=model,
                    max_tokens_before_summary=170000,
                    messages_to_keep=6,
                ),
                AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
                PatchToolCallsMiddleware(),
            ],
            default_interrupt_on=interrupt_on,
            general_purpose_agent=True,
        ),
        SummarizationMiddleware(
            model=model,
            max_tokens_before_summary=170000,
            messages_to_keep=6,
        ),
        AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
        PatchToolCallsMiddleware(),
    ]
    if interrupt_on is not None:
        deepagent_middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))
    if middleware is not None:
        deepagent_middleware.extend(middleware)

    graph = create_agent(
        model,
        system_prompt=system_prompt + "\n\n" + BASE_AGENT_PROMPT if system_prompt else BASE_AGENT_PROMPT,
        tools=tools,
        middleware=deepagent_middleware,
        response_format=response_format,
        context_schema=context_schema,
        checkpointer=checkpointer,
        store=store,
        debug=debug,
        name=name,
        cache=cache,
    ).with_config({"recursion_limit": 1000})

    # Wrap in DeepAgent if sandbox was provided
    if sandbox_backend is not None:
        return DeepAgent(graph, sandbox=sandbox_backend)

    return graph
