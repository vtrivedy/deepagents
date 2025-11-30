# 🏗️ DeepAgents: Complete Architecture Deep Dive

> A conversational walkthrough of every component, pattern, and design decision in the deepagents framework

## Table of Contents
1. [The Big Picture](#the-big-picture)
2. [Core Primitives](#core-primitives)
3. [Backend Architecture](#backend-architecture)
4. [Middleware Architecture](#middleware-architecture)
5. [Agent Construction](#agent-construction)
6. [Tool Integration](#tool-integration)
7. [CLI Integration Patterns](#cli-integration-patterns)
8. [Advanced Patterns](#advanced-patterns)
9. [Design Decisions & Tradeoffs](#design-decisions)

---

## The Big Picture

### What Problem Does This Solve?

Traditional LLM applications look like this:

```
User → Prompt → LLM → Response
```

But for complex tasks like "analyze this codebase and refactor the authentication system", you need:

```
User → Agent → [
    Plan → Read Files → Analyze → Create Plan →
    Delegate to Subagent → Review → Execute Changes →
    Test → Iterate
] → Response
```

This involves **50-100+ tool calls** across **hours** of execution. The challenges:

1. **Context Overflow**: Each tool call adds to context. At 100 calls × 1000 tokens = 100k tokens
2. **Cost**: GPT-4 at $30/1M tokens = $3 per complex task
3. **Reliability**: Long chains fail. One bad tool call breaks everything
4. **Observability**: Hard to debug 100-step chains
5. **State Management**: Need to checkpoint progress for resume

### The DeepAgents Solution

DeepAgents uses **3 core patterns**:

1. **Planning** (TodoListMiddleware): Break tasks into trackable steps
2. **Context Offloading** (FilesystemMiddleware): Save large outputs to files instead of context
3. **Delegation** (SubAgentMiddleware): Isolate subtasks in separate contexts

Plus **extensibility patterns**:
4. **Pluggable Backends**: Swap storage (in-memory, disk, remote sandbox)
5. **Middleware Architecture**: Inject tools/prompts/hooks without changing core logic
6. **HITL Interrupts**: Pause for human approval on dangerous operations

---

## Core Primitives

Every abstraction in deepagents builds on these primitives:

### Primitive 1: BackendProtocol

**What**: The interface for "where files live"

**Why**: Decouple agent logic from storage. Same agent code works with:
- In-memory state (testing/prototyping)
- Local filesystem (development)
- Remote sandbox (production)
- Database (persistence)

**The Interface**:
```python
@runtime_checkable
class BackendProtocol(Protocol):
    def ls_info(self, path: str) -> list[FileInfo]: ...
    def read(self, file_path: str, offset: int, limit: int) -> str: ...
    def write(self, file_path: str, content: str) -> WriteResult: ...
    def edit(self, file_path: str, old: str, new: str, replace_all: bool) -> EditResult: ...
    def glob_info(self, pattern: str, path: str) -> list[FileInfo]: ...
    def grep_raw(self, pattern: str, path: str | None, glob: str | None) -> list[GrepMatch]: ...
```

**Key Design Decision**: Files are represented as dicts:
```python
{
    "content": ["line 1", "line 2", ...],  # List of strings (not joined!)
    "created_at": "2025-01-15T10:00:00Z",
    "modified_at": "2025-01-15T10:05:00Z"
}
```

Why list of strings? **Efficient line-based operations**:
- Pagination: `content[offset:offset+limit]`
- Editing: Find line, replace, no need to split/join
- Grep: Iterate lines with enumerate()

**Extended Protocol: SandboxBackendProtocol**

For remote execution environments:
```python
class SandboxBackendProtocol(BackendProtocol, Protocol):
    def execute(self, command: str) -> ExecuteResponse: ...

    @property
    def id(self) -> str:  # Unique sandbox identifier
        ...
```

### Primitive 2: AgentMiddleware

**What**: Lifecycle hooks for extending agents

**Why**: Add capabilities without modifying agent core. Think of it like Django middleware or Express.js middleware.

**The Interface**:
```python
class AgentMiddleware:
    # Tool injection
    tools: list[BaseTool] = []

    # State schema extension
    state_schema: type[AgentState] | None = None

    # Lifecycle hooks
    def before_agent(self, state, runtime) -> dict: ...
    def after_agent(self, state, runtime) -> dict: ...
    def wrap_model_call(self, request, handler) -> response: ...
    # ... and more
```

**Execution Order**:
```
Request comes in
  ↓
before_agent() hooks (in order)
  ↓
Agent execution
  ↓
  ├─> wrap_model_call() hooks (in order, like onion)
  │     ├─> Modify request
  │     ├─> Call next middleware
  │     └─> Modify response
  ↓
after_agent() hooks (in reverse order)
  ↓
Response returned
```

### Primitive 3: State (TypedDict)

**What**: The data structure flowing through the agent

**Base State**:
```python
class AgentState(TypedDict):
    messages: list[BaseMessage]  # Conversation history
```

**Extended by Middleware**:
```python
class TodoState(AgentState):
    todos: list[dict]  # From TodoListMiddleware

class FilesystemState(AgentState):
    files: dict[str, FileData]  # From FilesystemMiddleware

class SubAgentState(AgentState):
    subagent_results: dict  # From SubAgentMiddleware
```

**Key Pattern**: Middleware extends state through `state_schema`:
```python
class TodoListMiddleware(AgentMiddleware):
    state_schema = TodoState  # Automatically merged into agent state
```

---

## Backend Architecture

Let's explore each backend implementation to see the pattern in action.

### StateBackend: In-Memory State

**Use Case**: Default. Files stored in LangGraph state (checkpointed but ephemeral per thread)

**Key Implementation Detail**:
```python
class StateBackend(BackendProtocol):
    def __init__(self, runtime: ToolRuntime):
        self.runtime = runtime  # Access to agent state

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        files = self.runtime.state.get("files", {})
        file_data = files.get(file_path)

        if not file_data:
            return f"Error: File '{file_path}' not found"

        lines = file_data["content"][offset:offset+limit]
        return format_with_line_numbers(lines, start=offset+1)
```

**When to Use**:
- ✅ Prototyping
- ✅ Testing
- ✅ Temporary workspaces
- ❌ Persistent storage (state is thread-scoped)
- ❌ Large files (state has size limits)

### FilesystemBackend: Real Disk Operations

**Use Case**: Development, local execution, working with real codebases

**Key Implementation**:
```python
class FilesystemBackend(BackendProtocol):
    def __init__(self, root_dir: str | Path = "/"):
        self.root = Path(root_dir).resolve()

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        physical_path = self.root / file_path.lstrip("/")

        try:
            with open(physical_path, "r") as f:
                lines = f.readlines()[offset:offset+limit]
            return format_with_line_numbers(lines, start=offset+1)
        except FileNotFoundError:
            return f"Error: File '{file_path}' not found"

    def write(self, file_path: str, content: str) -> WriteResult:
        physical_path = self.root / file_path.lstrip("/")

        # Check if file exists
        if physical_path.exists():
            return WriteResult(error=f"File {file_path} already exists")

        physical_path.parent.mkdir(parents=True, exist_ok=True)
        physical_path.write_text(content)

        # Note: files_update=None means "already persisted externally"
        return WriteResult(path=file_path, files_update=None)
```

**Key Difference from StateBackend**:
- `files_update=None` (not in state, already on disk)
- Real file I/O with error handling
- Path resolution: `/workspace/file.txt` → `{root_dir}/workspace/file.txt`

**When to Use**:
- ✅ Local development
- ✅ Working with existing codebases
- ✅ Large files
- ✅ Integration with git/build tools
- ❌ Remote execution (use SandboxBackend)

### StoreBackend: Persistent Cross-Thread Storage

**Use Case**: Long-term memory, knowledge bases, preferences

**Key Implementation**:
```python
class StoreBackend(BackendProtocol):
    def __init__(self, store: BaseStore, namespace: tuple[str, ...] = ("default",)):
        self.store = store  # LangGraph Store (Postgres, Redis, etc.)
        self.namespace = namespace

    def read(self, file_path: str, ...) -> str:
        item = self.store.get(self.namespace, file_path)
        if not item:
            return f"Error: File '{file_path}' not found"

        file_data = item.value
        lines = file_data["content"][offset:offset+limit]
        return format_with_line_numbers(lines, start=offset+1)

    def write(self, file_path: str, content: str) -> WriteResult:
        file_data = create_file_data(content)
        self.store.put(self.namespace, file_path, file_data)

        # files_update=None (persisted to store, not state)
        return WriteResult(path=file_path, files_update=None)
```

**When to Use**:
- ✅ User preferences (persist across all conversations)
- ✅ Knowledge bases (accumulate over time)
- ✅ Agent memory (learn from past interactions)
- ✅ Shared state (multiple agents accessing same data)

**Example: Long-term Memory**:
```python
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend

agent = create_deep_agent(
    backend=CompositeBackend(
        default=StateBackend(),  # Ephemeral workspace
        routes={
            "/memories/": StoreBackend(store=PostgresStore())  # Persistent
        }
    )
)

# Agent can:
# - write_file("/workspace/temp.py") → Goes to state (ephemeral)
# - write_file("/memories/preferences.md") → Goes to store (persisted)
```

### CompositeBackend: Routing Multiple Backends

**Use Case**: Hybrid storage (ephemeral + persistent)

**Implementation Pattern**:
```python
class CompositeBackend(BackendProtocol):
    def __init__(
        self,
        default: BackendProtocol,
        routes: dict[str, BackendProtocol]
    ):
        self.default = default
        self.routes = sorted(routes.items(), key=lambda x: -len(x[0]))  # Longest prefix first

    def _get_backend(self, path: str) -> BackendProtocol:
        for prefix, backend in self.routes:
            if path.startswith(prefix):
                return backend
        return self.default

    def read(self, file_path: str, ...) -> str:
        backend = self._get_backend(file_path)
        return backend.read(file_path, ...)
```

**Example: Multi-level Routing**:
```python
CompositeBackend(
    default=StateBackend(),  # Ephemeral workspace
    routes={
        "/memories/": StoreBackend(store),      # Long-term memory
        "/workspace/": FilesystemBackend("/"),   # Real filesystem
        "/cache/": RedisBackend(redis_client)    # Fast cache
    }
)

# Routing:
# /workspace/code.py → FilesystemBackend
# /memories/prefs.md → StoreBackend
# /cache/data.json → RedisBackend
# /temp/scratch.txt → StateBackend (default)
```

### SandboxBackend: Remote Execution

**Use Case**: Production, isolation, cloud execution

**Additional Capabilities**:
```python
class SandboxBackend(SandboxBackendProtocol):
    def execute(self, command: str) -> ExecuteResponse:
        """Run shell command in sandbox."""
        # Send command to remote sandbox (Modal, E2, etc.)
        result = self.sandbox_client.execute(command)

        return ExecuteResponse(
            output=result.stdout + result.stderr,
            exit_code=result.exit_code,
            truncated=len(result.output) > MAX_OUTPUT
        )

    @property
    def id(self) -> str:
        return self.sandbox_instance_id  # Unique per sandbox
```

**Example: Modal Integration**:
```python
from deepagents_cli.integrations.modal import ModalBackend

# Spins up a Modal container
sandbox = ModalBackend()

agent = create_deep_agent(backend=sandbox)

# Now agent can:
# - read_file("/workspace/code.py") → Reads from Modal container
# - execute("python test.py") → Runs in Modal container
# - write_file("/output/result.json") → Writes to Modal container
```

---

## Middleware Architecture

Middleware is the **extensibility mechanism**. Let's see how each middleware works.

### TodoListMiddleware: Task Planning

**What it does**:
1. Adds `write_todos(todos: list)` tool
2. Adds `read_todos()` tool
3. Extends state with `todos: list[dict]`
4. Injects system prompt about when/how to use todos

**Implementation Pattern**:
```python
@tool
def write_todos(todos: list[dict]) -> str:
    """Update the todo list."""
    # This tool returns a Command to update state
    return Command(update={"todos": todos})

class TodoListMiddleware(AgentMiddleware):
    tools = [write_todos, read_todos]
    state_schema = TodoState  # Adds todos to state

    def wrap_model_call(self, request, handler):
        # Inject prompt about todos
        todos_prompt = get_todo_system_prompt()
        request.system_prompt += "\n\n" + todos_prompt
        return handler(request)
```

**System Prompt Injection**:
```
## Task Management

When using the write_todos tool:
1. Keep the todo list MINIMAL - aim for 3-6 items maximum
2. Only create todos for complex, multi-step tasks that truly need tracking
3. Break down work into clear, actionable items without over-fragmenting
4. For simple tasks (1-2 steps), just do them directly without creating todos
5. When first creating a todo list for a task, ALWAYS ask the user if the plan looks good
   - Create the todos, let them render, then ask: "Does this plan look good?"
   - Wait for the user's response before marking the first todo as in_progress
   - If they want changes, adjust the plan accordingly
6. Update todo status promptly as you complete each item
```

**Usage Pattern**:
```python
# Agent reasoning:
# "This is a multi-step task. Let me break it down."

write_todos([
    {"content": "Research async patterns", "status": "in_progress"},
    {"content": "Implement async handler", "status": "pending"},
    {"content": "Write tests", "status": "pending"}
])

# Later:
write_todos([
    {"content": "Research async patterns", "status": "completed"},
    {"content": "Implement async handler", "status": "in_progress"},
    {"content": "Write tests", "status": "pending"}
])
```

### FilesystemMiddleware: File Operations

**What it does**:
1. Adds 6-7 file tools (ls, read, write, edit, glob, grep, execute*)
2. Extends state with `files: dict` (if using StateBackend)
3. Injects system prompt explaining file operations
4. Implements context offloading (saves large tool results to files)

**Tool Creation Pattern**:
```python
class FilesystemMiddleware(AgentMiddleware):
    def __init__(self, backend: BackendProtocol):
        self.backend = backend

        # Create tools dynamically
        self.tools = [
            self._make_ls_tool(),
            self._make_read_tool(),
            self._make_write_tool(),
            self._make_edit_tool(),
            self._make_glob_tool(),
            self._make_grep_tool(),
        ]

        # Add execute tool if backend supports it
        if isinstance(backend, SandboxBackendProtocol):
            self.tools.append(self._make_execute_tool())

    def _make_read_tool(self) -> BaseTool:
        @tool
        def read_file(file_path: str, offset: int = 0, limit: int = 2000) -> str:
            """Read file content with line numbers.

            Args:
                file_path: Absolute path (must start with /)
                offset: Line number to start from (0-indexed)
                limit: Max lines to read (default 2000)
            """
            return self.backend.read(file_path, offset, limit)

        return read_file
```

**Context Offloading Logic**:
```python
def after_tools(self, state, runtime):
    """Save large tool results to files to prevent context overflow."""
    last_message = state["messages"][-1]

    if isinstance(last_message, ToolMessage):
        content = last_message.content

        # If output is large, save to file
        if len(content) > 10000:  # 10k chars
            file_path = f"/tool_outputs/{last_message.tool_call_id}.txt"
            self.backend.write(file_path, content)

            # Replace message content with reference
            last_message.content = f"[Large output saved to {file_path}]"

    return {}  # No state update needed
```

**Why This Matters**: Without offloading, a 50-step task could accumulate 500k tokens in tool outputs, making the agent unusable!

### SubAgentMiddleware: Task Delegation

**What it does**:
1. Adds `task(description: str, subagent_type: str)` tool
2. Manages subagent lifecycle (create, execute, clean up)
3. Isolates context (subagent doesn't see parent's conversation)
4. Enables parallelization (multiple subagents can run concurrently)

**Implementation Skeleton**:
```python
class SubAgentMiddleware(AgentMiddleware):
    def __init__(
        self,
        default_model,
        default_tools,
        subagents: list[SubAgent],
        default_middleware,
        ...
    ):
        self.subagents = {s["name"]: s for s in subagents}
        self.default_config = {...}

        self.tools = [self._make_task_tool()]

    def _make_task_tool(self):
        @tool
        def task(description: str, subagent_type: str = "general-purpose") -> str:
            """Delegate a task to a specialized sub-agent.

            The sub-agent runs in isolation with its own context window.
            Use this for:
            - Complex subtasks that would clutter your context
            - Parallel work (research while coding)
            - Specialized expertise (data analysis, web research)
            """
            # Get subagent config
            config = self.subagents.get(subagent_type, self.default_config)

            # Create subagent graph
            subagent = create_deep_agent(
                model=config["model"],
                tools=config["tools"],
                system_prompt=config["system_prompt"],
                middleware=config["middleware"],
            )

            # Execute task
            result = subagent.invoke({
                "messages": [{"role": "user", "content": description}]
            })

            # Return just the final response
            return result["messages"][-1].content

        return task
```

**Usage Example**:
```python
# Main agent:
"I need to research async patterns and implement a handler. Let me parallelize."

# Spawn two subagents:
task(
    description="Research Python async best practices and summarize in 3 bullet points",
    subagent_type="research-agent"
)

task(
    description="Implement an async request handler with error handling",
    subagent_type="coding-agent"
)

# Main agent reconciles results and synthesizes final response
```

### SummarizationMiddleware: Context Management

**What it does**:
1. Monitors context size (counts tokens)
2. When exceeds threshold (170k tokens), triggers summarization
3. Summarizes old messages, keeps recent ones
4. Injects summary as system message

**Implementation**:
```python
class SummarizationMiddleware(AgentMiddleware):
    def __init__(self, model, max_tokens_before_summary: int = 170000):
        self.model = model
        self.threshold = max_tokens_before_summary

    def before_agent(self, state, runtime):
        messages = state["messages"]
        total_tokens = count_tokens(messages)

        if total_tokens > self.threshold:
            # Summarize old messages (keep last 6)
            to_summarize = messages[:-6]
            summary = self._summarize(to_summarize)

            # Update state
            return {
                "messages": [
                    SystemMessage(content=f"Previous conversation summary:\n{summary}"),
                    *messages[-6:]  # Keep recent messages
                ]
            }

        return {}  # No update needed

    def _summarize(self, messages):
        prompt = f"""Summarize this conversation concisely:

        {format_messages(messages)}

        Summary:"""

        return self.model.invoke(prompt).content
```

**Why This Matters**: Allows agents to work on tasks spanning 200+ tool calls without hitting context limits!

### HumanInTheLoopMiddleware: Safety & Control

**What it does**:
1. Configured via `interrupt_on` parameter
2. Pauses execution before dangerous tools
3. Waits for human approval/rejection
4. Resumes with decision

**Configuration**:
```python
agent = create_deep_agent(
    tools=[dangerous_tool],
    interrupt_on={
        "shell": {
            "allowed_decisions": ["approve", "reject"],
            "description": lambda tool_call, state, runtime:
                f"Run command: {tool_call['args']['command']}"
        }
    }
)
```

**Execution Flow**:
```
Agent wants to call shell("rm -rf /")
  ↓
HumanInTheLoopMiddleware intercepts
  ↓
Raises NodeInterrupt exception
  ↓
LangGraph pauses and emits __interrupt__ in stream
  ↓
CLI shows approval UI
  ↓
User approves/rejects
  ↓
CLI calls agent.invoke(Command(resume={"decisions": [...]}))
  ↓
Middleware receives decision
  ↓
If approved: Execute tool
If rejected: Return error message to agent
```

---

## Agent Construction: How It All Comes Together

The `create_deep_agent()` function is the **composition root** - where all primitives combine into a working agent.

### The Assembly Process

```python
def create_deep_agent(
    model=None,
    tools=None,
    system_prompt=None,
    middleware=(),
    subagents=None,
    backend=None,
    interrupt_on=None,
    ...
):
    # Step 1: Default model
    if model is None:
        model = ChatAnthropic(
            model_name="claude-sonnet-4-5-20250929",
            max_tokens=20000
        )

    # Step 2: Configure summarization thresholds
    if model.profile and "max_input_tokens" in model.profile:
        # Use model's known limits (e.g., Gemini with 1M tokens)
        trigger = ("fraction", 0.85)  # Summarize at 85% of limit
        keep = ("fraction", 0.10)     # Keep last 10% of messages
    else:
        # Conservative defaults (170k tokens, ~85k words)
        trigger = ("tokens", 170000)
        keep = ("messages", 6)  # Keep last 6 messages

    # Step 3: Assemble middleware stack
    deepagent_middleware = [
        # 1. TodoListMiddleware - Always first for planning
        TodoListMiddleware(),

        # 2. FilesystemMiddleware - File operations
        FilesystemMiddleware(backend=backend),

        # 3. SubAgentMiddleware - Task delegation
        SubAgentMiddleware(
            default_model=model,
            default_tools=tools,
            subagents=subagents or [],
            # Subagents get their own middleware stack!
            default_middleware=[
                TodoListMiddleware(),
                FilesystemMiddleware(backend=backend),
                SummarizationMiddleware(...),  # Nested summarization
                AnthropicPromptCachingMiddleware(),
                PatchToolCallsMiddleware(),
            ],
            default_interrupt_on=interrupt_on,
        ),

        # 4. SummarizationMiddleware - Context management
        SummarizationMiddleware(
            model=model,
            trigger=trigger,
            keep=keep,
        ),

        # 5. AnthropicPromptCachingMiddleware - Cost optimization (Anthropic only)
        AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),

        # 6. PatchToolCallsMiddleware - Cleanup dangling tool calls
        PatchToolCallsMiddleware(),
    ]

    # Step 4: Add custom middleware
    if middleware:
        deepagent_middleware.extend(middleware)

    # Step 5: Add HITL middleware last (runs first for tool interception)
    if interrupt_on is not None:
        deepagent_middleware.append(
            HumanInTheLoopMiddleware(interrupt_on=interrupt_on)
        )

    # Step 6: Delegate to LangChain's create_agent
    return create_agent(
        model,
        system_prompt=system_prompt + "\n\n" + BASE_AGENT_PROMPT,
        tools=tools,
        middleware=deepagent_middleware,
        checkpointer=checkpointer,
        store=store,
        ...
    ).with_config({"recursion_limit": 1000})
```

### Middleware Order Matters!

The order is carefully designed:

```
1. TodoListMiddleware        ← Adds planning tools
2. FilesystemMiddleware       ← Adds file tools
3. SubAgentMiddleware         ← Adds delegation tool
4. SummarizationMiddleware    ← Monitors context, triggers cleanup
5. PromptCachingMiddleware    ← Optimizes costs (Anthropic)
6. PatchToolCallsMiddleware   ← Fixes interrupted tool calls
7. Custom Middleware          ← Your extensions
8. HumanInTheLoopMiddleware   ← Intercepts dangerous tools (LAST!)
```

**Why this order?**

1. **Tool middleware first**: Todos, files, subagents provide capabilities agent needs
2. **Summarization in middle**: Can observe all tool calls and state
3. **Utility middleware**: Prompt caching, patches applied to final state
4. **Custom middleware**: Your code runs after defaults
5. **HITL last**: Must intercept tool calls BEFORE execution

**Example of order impact**:

```python
# BAD ORDER:
middleware = [
    HumanInTheLoopMiddleware(...),  # Runs first (wrapped innermost)
    TodoListMiddleware(),            # Runs after HITL
]

# Problem: HITL can't intercept write_todos because it doesn't exist yet!

# GOOD ORDER:
middleware = [
    TodoListMiddleware(),            # Adds write_todos
    HumanInTheLoopMiddleware(...),   # Can intercept write_todos
]
```

### Subagent Nesting

Notice subagents get their OWN middleware stack:

```python
SubAgentMiddleware(
    default_middleware=[
        TodoListMiddleware(),         # Subagents can plan
        FilesystemMiddleware(),       # Subagents can read/write
        SummarizationMiddleware(),    # Subagents can summarize
        # BUT no SubAgentMiddleware → No recursive delegation
    ]
)
```

This prevents infinite nesting (subagent → subagent → subagent...).

### Backend Injection

The backend flows through middleware:

```python
backend = FilesystemBackend(root_dir="/workspace")

middleware = [
    TodoListMiddleware(),                      # No backend needed
    FilesystemMiddleware(backend=backend),     # Uses backend for tools
    SubAgentMiddleware(
        default_middleware=[
            FilesystemMiddleware(backend=backend),  # Subagents share backend!
        ]
    ),
]
```

**Key insight**: All agents (main + subagents) share the same backend. They all operate on the same "filesystem"!

---

## Tool Integration Patterns

Now let's see how tools are created and integrated.

### Pattern 1: Simple Function Tool

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the weather in a city."""
    return f"The weather in {city} is sunny."

agent = create_deep_agent(tools=[get_weather])
```

**What happens**:
1. `@tool` decorator converts function to `StructuredTool`
2. Tool schema extracted from docstring + type hints
3. LLM sees tool in available actions
4. When called, function executes and returns result

### Pattern 2: Stateful Middleware Tool

```python
class WeatherMiddleware(AgentMiddleware):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.tools = [self._make_weather_tool()]

    def _make_weather_tool(self):
        @tool
        def get_weather(city: str) -> str:
            """Get weather using API."""
            # Closure captures self
            return requests.get(
                "https://api.weather.com",
                params={"city": city, "key": self.api_key}
            ).json()

        return get_weather

agent = create_deep_agent(middleware=[WeatherMiddleware(api_key="...")])
```

**Why closure?** Tool needs access to middleware state (api_key, backend, etc.)

### Pattern 3: Backend-Powered Tool

How FilesystemMiddleware creates tools:

```python
class FilesystemMiddleware(AgentMiddleware):
    def __init__(self, backend: BackendProtocol):
        self.backend = backend
        self.tools = [
            self._make_read_tool(),
            self._make_write_tool(),
            # ...
        ]

    def _make_read_tool(self):
        # Closure captures self.backend
        backend = self.backend

        @tool
        def read_file(file_path: str, offset: int = 0, limit: int = 2000) -> str:
            """Read file with pagination.

            Args:
                file_path: Absolute path (must start with /)
                offset: Line offset (0-indexed)
                limit: Max lines to read
            """
            return backend.read(file_path, offset, limit)

        return read_file
```

**Key pattern**: Tool is a closure over backend. Same tool definition works with any backend!

### Pattern 4: State-Updating Tool

Tools that need to update LangGraph state:

```python
from langgraph.types import Command

class TodoListMiddleware(AgentMiddleware):
    def _make_write_todos_tool(self):
        @tool
        def write_todos(todos: list[dict]) -> str:
            """Update the todo list.

            Args:
                todos: List of todo items with content, status, activeForm
            """
            # Don't return normal response - return Command!
            return Command(update={"todos": todos})

        return write_todos
```

**Why Command?** LangGraph state is immutable - you can't mutate it directly. `Command(update={...})` tells LangGraph to merge this into state.

### Pattern 5: Tool with Runtime Access

Some tools need access to the agent's runtime (state, config, etc.):

```python
from langchain.tools import ToolRuntime

@tool
def read_file(
    file_path: str,
    runtime: ToolRuntime,  # Injected automatically!
) -> str:
    """Read a file."""
    # Access current state
    files = runtime.state.get("files", {})

    # Access config
    thread_id = runtime.config["configurable"]["thread_id"]

    # Read file
    return files.get(file_path, "File not found")
```

**How it works**: LangChain detects the `runtime: ToolRuntime` parameter and injects it automatically. The LLM never sees this parameter!

---

## CLI Integration Patterns

Now let's see how the CLI (deepagents-cli) uses the core library.

### CLI Architecture

```
libs/deepagents-cli/
├── deepagents_cli/
│   ├── main.py              # Entry point, CLI loop
│   ├── config.py            # Settings, project detection
│   ├── agent.py             # Agent creation wrapper
│   ├── execution.py         # Task execution, streaming
│   ├── input.py             # Prompt toolkit input handling
│   ├── ui.py                # Rich output rendering
│   ├── file_ops.py          # File operation tracking, diffs
│   ├── commands.py          # Slash commands (/help, /clear)
│   ├── tools.py             # CLI-specific tools (web_search, fetch_url)
│   ├── agent_memory.py      # Memory middleware
│   ├── shell.py             # Shell middleware (local mode)
│   └── skills/              # Skills system
│       └── middleware.py
```

### Pattern 1: Settings & Environment Detection

The CLI detects the environment at startup:

```python
# config.py
@dataclass
class Settings:
    openai_api_key: str | None
    anthropic_api_key: str | None
    tavily_api_key: str | None
    project_root: Path | None  # Detected via .git directory

    @classmethod
    def from_environment(cls) -> "Settings":
        # Detect API keys
        openai_key = os.environ.get("OPENAI_API_KEY")
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        tavily_key = os.environ.get("TAVILY_API_KEY")

        # Detect project root (walk up tree looking for .git)
        project_root = _find_project_root(Path.cwd())

        return cls(
            openai_api_key=openai_key,
            anthropic_api_key=anthropic_key,
            tavily_api_key=tavily_key,
            project_root=project_root,
        )

# Global instance
settings = Settings.from_environment()
```

**Usage**:
```python
if settings.has_tavily:
    tools.append(web_search_tool)

if settings.has_project:
    project_skills_dir = settings.project_root / ".deepagents" / "skills"
```

### Pattern 2: Hierarchical Memory

The CLI implements a 2-level memory system:

```
~/.deepagents/agent/agent.md          ← User memory (global)
~/project/.deepagents/agent.md        ← Project memory (local)
```

Both are loaded and combined:

```python
class AgentMemoryMiddleware(AgentMiddleware):
    def before_agent(self, state, runtime):
        result = {}

        # Load user memory
        user_path = settings.get_user_agent_md_path(assistant_id)
        if user_path.exists():
            result["user_memory"] = user_path.read_text()

        # Load project memory
        if settings.has_project:
            project_path = settings.get_project_agent_md_path()
            if project_path and project_path.exists():
                result["project_memory"] = project_path.read_text()

        return result

    def wrap_model_call(self, request, handler):
        # Inject both into system prompt
        system_prompt = f"""
        <user_memory>
        {request.state.get('user_memory', '')}
        </user_memory>

        <project_memory>
        {request.state.get('project_memory', '')}
        </project_memory>

        {request.system_prompt}
        """

        return handler(request.override(system_prompt=system_prompt))
```

**Why two levels?**
- User memory: Personal preferences across all projects
- Project memory: Project-specific conventions, architecture

### Pattern 3: Local vs Sandbox Mode

The CLI supports two execution modes:

**Local Mode** (no sandbox):
```python
# Uses local filesystem + shell
backend = FilesystemBackend()  # Real disk operations
middleware = [
    AgentMemoryMiddleware(),
    ShellMiddleware(),  # Provides shell tool for local execution
]
```

**Sandbox Mode** (remote execution):
```python
# Uses remote sandbox
from deepagents_cli.integrations.modal import ModalBackend

backend = ModalBackend()  # Remote Modal container
middleware = [
    AgentMemoryMiddleware(),
    # No ShellMiddleware - execute tool comes from backend
]
```

The key difference:

```python
def create_agent_with_config(
    model, assistant_id, tools,
    sandbox=None,  # Optional sandbox backend
):
    if sandbox is None:
        # LOCAL MODE
        backend = FilesystemBackend()
        middleware = [
            AgentMemoryMiddleware(...),
            ShellMiddleware(...),  # Local shell
        ]
    else:
        # SANDBOX MODE
        backend = sandbox  # Modal, E2, etc.
        middleware = [
            AgentMemoryMiddleware(...),
            # No shell - backend provides execute tool
        ]

    agent = create_deep_agent(
        model=model,
        backend=backend,
        middleware=middleware,
        ...
    )

    return agent, backend
```

### Pattern 4: Streaming with Dual Mode

The CLI uses dual-mode streaming for real-time UI updates:

```python
async for chunk in agent.astream(
    input,
    stream_mode=["messages", "updates"],  # Both!
    subgraphs=True,
    config=config,
):
    namespace, mode, data = chunk

    if mode == "messages":
        # AI responses, tool calls, tool results
        message, metadata = data
        if isinstance(message, AIMessageChunk):
            # Stream text to terminal
            print(message.content, end="", flush=True)

    elif mode == "updates":
        # State updates, interrupts, todos
        if "__interrupt__" in data:
            # HITL approval needed
            decision = prompt_user_approval(data["__interrupt__"])
            agent.invoke(Command(resume=decision))

        if "todos" in data:
            # Render todo list
            render_todo_list(data["todos"])
```

**Why both streams?**
- `messages`: Real-time text streaming
- `updates`: State changes, interrupts, todos

### Pattern 5: File Operation Tracking

The CLI tracks file operations for beautiful diffs:

```python
class FileOpTracker:
    def start_operation(self, tool_name, args, tool_call_id):
        """Called when tool call is made."""
        if tool_name in {"write_file", "edit_file"}:
            # Capture "before" state
            path = args["file_path"]
            before = backend.download_files([path])[0].content

            self.active[tool_call_id] = FileOperationRecord(
                tool_name=tool_name,
                path=path,
                before_content=before,
            )

    def complete_with_message(self, tool_message):
        """Called when tool result arrives."""
        record = self.active.get(tool_message.tool_call_id)

        # Capture "after" state
        after = backend.download_files([record.path])[0].content

        # Compute diff
        record.diff = compute_unified_diff(
            record.before_content,
            after,
            context_lines=3,
        )

        record.metrics.lines_added = count_additions(record.diff)
        record.metrics.lines_removed = count_deletions(record.diff)

        # Display
        render_file_operation(record)
```

This enables rich display:
```
✏️ write_file(config.py)
  ⎿  Created 45 lines · 1234 bytes

[Syntax-highlighted diff shown here]
```

---

## Advanced Patterns

### Pattern 1: Composite Backends with Smart Routing

Combine multiple backends for hybrid storage:

```python
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend, FilesystemBackend

agent = create_deep_agent(
    backend=CompositeBackend(
        default=StateBackend(),  # Ephemeral workspace
        routes={
            "/memories/": StoreBackend(store=PostgresStore()),  # Long-term
            "/workspace/": FilesystemBackend("/real/path"),     # Real disk
            "/cache/": RedisBackend(redis_client),              # Fast cache
        }
    )
)

# Agent operations:
# write_file("/workspace/code.py")    → Real filesystem
# write_file("/memories/prefs.md")    → Postgres (persists)
# write_file("/cache/temp.json")      → Redis (fast)
# write_file("/scratch/tmp.txt")      → State (ephemeral)
```

### Pattern 2: Custom Subagent with Specialized Tools

```python
from tavily import TavilyClient

tavily = TavilyClient(api_key="...")

@tool
def deep_research(query: str, max_results: int = 10) -> str:
    """Deep web research with multiple sources."""
    results = tavily.search(query, max_results=max_results, include_raw_content=True)
    # Process and synthesize
    return synthesize_results(results)

research_subagent = {
    "name": "deep-researcher",
    "description": "Conducts in-depth research with web access",
    "system_prompt": """You are a research specialist.

    Your workflow:
    1. Use deep_research to gather comprehensive sources
    2. Cross-reference multiple sources
    3. Synthesize findings into structured report
    4. Cite all sources
    """,
    "tools": [deep_research],
    "model": "openai:gpt-4o",  # Different model than main agent
}

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4",  # Main agent
    subagents=[research_subagent],
)

# Main agent can now:
# task(
#     description="Research the latest advances in quantum computing",
#     subagent_type="deep-researcher"
# )
```

### Pattern 3: Progressive Context Offloading

Automatically save large tool outputs to files:

```python
class ContextOffloadingMiddleware(AgentMiddleware):
    def __init__(self, backend, threshold: int = 10000):
        self.backend = backend
        self.threshold = threshold

    def after_tools(self, state, runtime):
        messages = state["messages"]
        last_msg = messages[-1]

        if isinstance(last_msg, ToolMessage):
            content = last_msg.content

            # If output is large, save to file
            if len(content) > self.threshold:
                # Generate unique path
                file_path = f"/tool_outputs/{last_msg.tool_call_id}.txt"

                # Save to backend
                self.backend.write(file_path, content)

                # Replace message content
                last_msg.content = f"""[Large output saved to {file_path}]

To view: read_file('{file_path}')

Summary:
{content[:500]}...
"""

        return {}  # No state update needed
```

This prevents context overflow on long tasks!

### Pattern 4: Multi-Agent Coordination

Parallel subagents with result synthesis:

```python
# Main agent reasoning:
# "I need to research AND code in parallel. Let me delegate."

# Spawn two subagents concurrently
research_result = task(
    description="Research Python async best practices. Focus on error handling.",
    subagent_type="research-agent"
)

code_result = task(
    description="Implement async request handler with retry logic",
    subagent_type="coding-agent"
)

# Main agent then synthesizes:
# "Based on research findings: {research_result}
#  And implementation: {code_result}
#  Here's the final solution..."
```

**Note**: Subagents run sequentially by default. For true parallel execution, you'd need to use LangGraph's built-in parallelization or external orchestration.

### Pattern 5: Skills System (CLI-specific)

The CLI adds a "skills" system - reusable scripts/workflows:

```
~/.deepagents/agent/skills/
└── web-research/
    ├── skill.md          # Skill description
    └── script.py         # Optional implementation

~/project/.deepagents/skills/
└── analyze-logs/
    ├── skill.md
    └── parse.py
```

**How it works**:

1. Skills are discovered at startup
2. Injected into agent's system prompt:
   ```
   ## Available Skills

   You have access to these skills:

   - web-research: Research a topic using web search and synthesis
     Location: ~/.deepagents/agent/skills/web-research/
     To use: Read the skill.md for instructions

   - analyze-logs: Parse and analyze application logs
     Location: ~/project/.deepagents/skills/analyze-logs/
     To use: Run the parse.py script
   ```

3. Agent can read skill files and execute scripts:
   ```python
   # Agent reasoning:
   # "User wants log analysis. I have a skill for this!"

   read_file("~/.deepagents/agent/skills/analyze-logs/skill.md")
   # → Gets step-by-step instructions

   shell("python ~/.deepagents/agent/skills/analyze-logs/parse.py --input logs.txt")
   # → Executes the skill's script
   ```

**Why skills?** Reusable, shareable workflows that agents can discover and use!

---

## Design Decisions & Tradeoffs

Let's examine the key architectural decisions and their implications.

### Decision 1: Middleware vs Inheritance

**Choice**: Composition via middleware instead of subclassing

**Rationale**:
- ✅ Easy to mix and match capabilities
- ✅ No diamond inheritance problems
- ✅ Community can share middleware
- ❌ More complex than simple subclassing
- ❌ Execution order can be subtle

**Example**:
```python
# BAD: Inheritance approach
class TodoAgent(BaseAgent):
    def add_todos(self): ...

class FilesystemAgent(BaseAgent):
    def read_file(self): ...

class TodoFilesystemAgent(TodoAgent, FilesystemAgent):  # Multiple inheritance!
    pass

# GOOD: Middleware approach
agent = create_deep_agent(middleware=[
    TodoListMiddleware(),
    FilesystemMiddleware(),
])
```

### Decision 2: Protocol vs ABC

**Choice**: Use `Protocol` (structural typing) instead of `ABC` (nominal typing)

**Rationale**:
- ✅ Duck typing - any object with right methods works
- ✅ No need to explicitly inherit
- ✅ Easier testing (mocks don't need inheritance)
- ❌ Less explicit (no inheritance tree)
- ❌ Runtime checks slower

**Example**:
```python
# With Protocol
class MyCustomBackend:  # No inheritance needed!
    def read(self, path): ...
    def write(self, path, content): ...
    # ... other methods

backend = MyCustomBackend()
agent = create_deep_agent(backend=backend)  # Just works!

# With ABC, would need:
class MyCustomBackend(BackendProtocol):  # Explicit inheritance
    ...
```

### Decision 3: State in Dict vs Dataclass

**Choice**: State is `TypedDict` not `@dataclass`

**Rationale**:
- ✅ LangGraph uses dicts for state
- ✅ Easy to merge states from multiple middleware
- ✅ JSON-serializable by default
- ❌ No IDE autocomplete for keys
- ❌ No runtime validation

**Example**:
```python
# With TypedDict
class AgentState(TypedDict):
    messages: list[BaseMessage]
    todos: list[dict]  # IDE knows this exists

state = {"messages": [], "todos": []}
state["todos"].append(...)  # Works, but no autocomplete

# With dataclass would be:
@dataclass
class AgentState:
    messages: list[BaseMessage]
    todos: list[dict]

state = AgentState(messages=[], todos=[])
state.todos.append(...)  # Full IDE support!

# But harder to merge states from different middleware
```

### Decision 4: Files as Lists of Strings

**Choice**: File content is `list[str]` not single `str`

**Rationale**:
- ✅ Efficient line-based operations (pagination, grep)
- ✅ No need to split/join repeatedly
- ✅ Line numbers implicit (via index)
- ❌ Must join for full content
- ❌ Less intuitive than single string

**Example**:
```python
# With list of strings
file_data = {
    "content": ["line 1", "line 2", "line 3"]
}

# Pagination
page = file_data["content"][offset:offset+limit]  # O(1)

# Editing line 2
file_data["content"][1] = "new line 2"  # O(1)

# With single string
file_data = {
    "content": "line 1\nline 2\nline 3"
}

# Pagination
lines = file_data["content"].split("\n")  # O(n)
page = lines[offset:offset+limit]  # Then O(1)

# Editing line 2
lines = file_data["content"].split("\n")  # O(n)
lines[1] = "new line 2"
file_data["content"] = "\n".join(lines)  # O(n)
```

### Decision 5: Subagents Get Subset of Middleware

**Choice**: Subagents don't have SubAgentMiddleware (no recursive delegation)

**Rationale**:
- ✅ Prevents infinite nesting
- ✅ Simpler mental model
- ✅ Forces thinking about task breakdown
- ❌ Subagents can't delegate further
- ❌ No multi-level task hierarchies

**Alternative design**: Allow recursive delegation with depth limit
```python
SubAgentMiddleware(max_depth=3)  # Allow 3 levels
```

But current design is simpler!

### Decision 6: HITL Middleware Last

**Choice**: HumanInTheLoopMiddleware must be last in middleware stack

**Rationale**:
- ✅ Can intercept ALL tools (including from other middleware)
- ✅ Single point of control
- ❌ Must be added last (easy to forget)
- ❌ Order dependency (fragile)

**Why not always include it?** Optional feature - not all use cases need HITL.

### Decision 7: Backend Factory Pattern

**Choice**: Backend can be instance OR factory function

**Rationale**:
- ✅ Flexibility: `FilesystemBackend()` or `lambda rt: StateBackend(rt)`
- ✅ Lazy initialization (useful for runtime-dependent backends)
- ❌ Two ways to do the same thing (confusing)

**Example**:
```python
# Direct instance
agent = create_deep_agent(backend=FilesystemBackend())

# Factory (needed for StateBackend which needs runtime)
agent = create_deep_agent(backend=lambda rt: StateBackend(rt))
```

---

## Summary: The Core Patterns

After this deep dive, here are the 10 key patterns you should internalize:

1. **Backend Protocol**: Abstraction for "where files live" (state, disk, database, sandbox)
2. **Middleware Composition**: Capabilities added via middleware, not inheritance
3. **State Extension**: Each middleware extends state with new fields
4. **Tool Closures**: Tools capture backend/config via closures
5. **Lifecycle Hooks**: `before_agent`, `after_agent`, `wrap_model_call` for extension points
6. **Progressive Offloading**: Save large outputs to files to prevent context overflow
7. **Hierarchical Memory**: User-level + project-level memory files
8. **Dual-mode Streaming**: `messages` for content, `updates` for state/interrupts
9. **Backend Routing**: CompositeBackend routes paths to different storage
10. **Subagent Isolation**: Each subagent has its own context, preventing cross-contamination

These patterns enable:
- ✅ Long-horizon tasks (100+ tool calls)
- ✅ Cost optimization (context offloading, prompt caching, summarization)
- ✅ Reliability (HITL, checkpointing, error handling)
- ✅ Extensibility (custom middleware, tools, backends)
- ✅ Observability (streaming, file tracking, todo lists)

---

## Next Steps

To master deepagents:

1. **Build a simple agent**: Start with `create_deep_agent(tools=[...])`
2. **Add custom middleware**: Implement `AgentMiddleware` with your own tools
3. **Experiment with backends**: Try StateBackend → FilesystemBackend → CompositeBackend
4. **Create subagents**: Define specialized subagents for different tasks
5. **Integrate HITL**: Add `interrupt_on` for dangerous operations
6. **Study the CLI**: See how deepagents-cli builds a complete TUI on top

The architecture is designed to be learned incrementally - you can start simple and add complexity as needed!

