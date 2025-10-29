# Sandbox Backend for DeepAgents

Persistent sandbox environment for running agent code in isolated containers.

## Quick Start

```bash
# 1. Install sandbox provider (using Modal as example)
cd deepagents
uv pip install modal

# 2. Authenticate with provider
modal setup

# 3. Run example
cd ..
python3 modal_sandbox_demo.py
```

## What Is This?

A sandbox provider that gives your agent a persistent, isolated execution environment:

- **Persistent container** - Same sandbox for entire agent session
- **Lazy creation** - Sandbox created automatically on first file operation
- **Auto cleanup** - Agent manages sandbox lifecycle
- **File uploads** - Upload files from local filesystem
- **Multi-filesystem** - Use both local (in-memory) and sandbox (Modal) storage
- **Auto bash tool** - Command execution automatically enabled when sandbox supports it

## Architecture

Sandbox providers implement the `BackendProtocol` directly - no wrapper needed!

```
SandboxConfig(provider="modal")
    ↓
create_deep_agent() OR create_sandbox_provider()
    ↓
ModalSandboxProvider (implements BackendProtocol directly!)
    ├── ls_info, read, write, edit, grep_raw, glob_info (BackendProtocol)
    └── execute() (sandbox-specific bonus method)
    ↓
FilesystemMiddleware detects execute() → auto-generates bash tool
    ↓
agent.invoke() → sandbox created lazily on first /workspace/ operation
    ↓
agent.cleanup() OR context manager → async cleanup handled automatically
```

## Basic Usage

### Agent with Sandbox

```python
from deepagents import create_deep_agent
from deepagents.backends.sandbox import SandboxConfig

# Create agent with sandbox
agent = create_deep_agent(
    sandbox=SandboxConfig(provider="modal"),
    system_prompt="You are a coding assistant. Use /workspace/ for code."
)

# Use agent (sandbox created lazily on first file operation)
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Write a Python script to /workspace/hello.py"
    }]
})

print(result["messages"][-1].content)

# Cleanup when done
agent.cleanup()
```

### With Context Manager (Recommended)

```python
# Cleanup happens automatically
with create_deep_agent(sandbox=SandboxConfig(provider="modal")) as agent:
    result = agent.invoke({"messages": [...]})
    print(result["messages"][-1].content)
# Sandbox terminated here
```

### Direct Provider Access (No Agent)

```python
from deepagents.backends.sandbox import (
    SandboxConfig,
    create_sandbox_provider,
)

# Create sandbox provider
config = SandboxConfig(provider="modal")

# Use context manager for auto-cleanup
with create_sandbox_provider(config) as provider:
    # Get filesystem interface
    fs = provider.fs

    # Upload file
    fs.upload_file(file=b"print('hello')", path="/workspace/script.py")

    # Read file
    content = fs.read_text("/workspace/script.py")
    print(content)

    # Edit file
    result = fs.edit(
        path="/workspace/script.py",
        old_string="hello",
        new_string="goodbye"
    )

    # Execute command (provider-specific method)
    result = provider.execute("python /workspace/script.py", cwd="/workspace")
    print(result["stdout"])

    # List files
    files = fs.list("/workspace/")
# Cleanup automatic with context manager
```

## Configuration

```python
from deepagents.backends.sandbox import SandboxConfig, BootstrapConfig

config = SandboxConfig(
    provider="modal",

    # Container settings
    image="python:3.11-slim",
    timeout_seconds=1800,  # 30 minutes
    memory_mb=2048,        # 2GB RAM
    cpu_count=1.0,         # 1 CPU

    # Bootstrap (optional) - runs before first operation
    bootstrap=BootstrapConfig(
        # Clone git repo
        git_repo="https://github.com/user/repo.git",
        git_branch="main",

        # Copy local files into sandbox
        local_files={
            "./config.json": "/workspace/config.json",
            "./utils.py": "/workspace/utils.py"
        },

        # Run setup commands
        setup_script="pip install -r requirements.txt",
        workdir="/workspace"
    )
)
```

## File Operations

### Upload Files

```python
# Method 1: Upload from local file
backend.upload_file(
    local_path="./data.csv",
    path="/workspace/data.csv"
)

# Method 2: Upload bytes
backend.upload_file(
    content=b"binary data",
    path="/workspace/file.bin"
)

# Method 3: Upload string
backend.upload_file(
    content='{"key": "value"}',
    path="/workspace/config.json"
)

# Method 4: Using .fs interface
fs = backend.fs
fs.upload_file(file=b"content", path="/workspace/file.txt")
```

### Read, Edit, List

```python
# Read file
content = fs.read("/workspace/file.txt")

# Edit file
result = fs.edit(
    path="/workspace/file.py",
    old_string="old_value",
    new_string="new_value",
    replace_all=True
)

# List directory
files = fs.list("/workspace/")
for f in files:
    print(f"{f.path} - {f.size} bytes")
```

## Multi-Filesystem Support

Route different paths to different backends. DeepAgents supports two patterns:

### Pattern 1: Local Primary + Sandbox for /workspace/

This is the default when using `sandbox=` parameter. Fast local storage for notes, sandbox for code execution.

```python
from deepagents import create_deep_agent
from deepagents.backends.sandbox import SandboxConfig

agent = create_deep_agent(
    sandbox=SandboxConfig(provider="modal"),
    system_prompt="""You are a coding assistant.

File locations:
- /workspace/ - Sandbox for code execution (persistent during session)
- / - Quick notes (in-memory, fast, free)

Use /workspace/ for code, / for planning."""
)

# Agent can use both filesystems:
# - /notes.txt → Local in-memory (StateBackend) - ephemeral
# - /workspace/code.py → Modal sandbox - persistent during session
```

### Pattern 2: Modal Primary + Persistent /memory/

Use Modal sandbox as the **primary** backend for everything, with persistent storage for specific paths.

```python
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend
from deepagents.backends.store import StoreBackend
from deepagents.backends.sandbox import create_sandbox_provider, SandboxConfig
from langgraph.store.memory import InMemoryStore

# Create Modal provider (will be primary backend)
sandbox_config = SandboxConfig(provider="modal")
memory_store = InMemoryStore()

# Use context manager for automatic cleanup
with create_sandbox_provider(sandbox_config) as modal_provider:
    def create_backend(runtime):
        return CompositeBackend(
            default=modal_provider,  # ← Modal is PRIMARY! All paths → sandbox
            routes={
                "/memory/": StoreBackend(runtime)  # Only /memory/ persists across sessions
            }
        )

    agent = create_deep_agent(
        backend=create_backend,
        store=memory_store,  # Required for StoreBackend
        system_prompt="""You are a coding assistant.

File locations:
- / and /workspace/ → Modal sandbox (ephemeral, deleted on cleanup)
- /memory/ → Persistent storage (survives across agent sessions)

Use bash tool to run code."""
    )

    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": "Save 'Claude' to /memory/name.txt, create /workspace/hello.py, run it"
        }]
    })
    print(result["messages"][-1].content)
# ← Cleanup happens automatically here via context manager
```

**Key Differences:**
- Pattern 1: Local fast, sandbox isolated - best for most use cases
- Pattern 2: Everything in sandbox by default, /memory/ persists - best when you want full sandbox isolation + long-term memory

## Bash Tool Auto-Generation

When a backend supports command execution (has an `execute()` method), the `FilesystemMiddleware` automatically generates a `bash` tool for the agent. You don't need to create it manually!

### How It Works

```python
from deepagents import create_deep_agent
from deepagents.backends.sandbox import SandboxConfig

# The bash tool is created automatically when sandbox is provided
agent = create_deep_agent(
    sandbox=SandboxConfig(provider="modal"),
    system_prompt="You are a coding assistant. Use bash tool to run commands."
)

# Agent now has these tools automatically:
# - write_file, read_file, edit_file, ls, glob_search, grep_search (filesystem tools)
# - bash (auto-generated for sandbox execution)
```

### Detection Logic

The middleware checks if the backend has an `execute()` method:

```python
# FilesystemMiddleware internals (you don't write this)
def _backend_supports_execution(backend) -> bool:
    if callable(backend):
        return True  # Conservative - include bash tool
    return hasattr(backend, 'execute') and callable(getattr(backend, 'execute', None))
```

### Bash Tool Usage

The agent can use the bash tool like any other tool:

```python
# Agent invokes bash tool automatically when needed:
agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Create hello.py in /workspace/ and run it with bash"
    }]
})

# Agent will:
# 1. Use write_file to create /workspace/hello.py
# 2. Use bash tool: bash(command="python /workspace/hello.py")
```

### Manual Bash Tool (Advanced)

If you need custom bash tool behavior, you can provide your own:

```python
from langchain_core.tools import tool

@tool
def custom_bash(command: str) -> str:
    """Execute bash command with custom logic"""
    # Your custom implementation
    result = provider.execute(command, cwd="/custom/path")
    return result["stdout"]

agent = create_deep_agent(
    sandbox=SandboxConfig(provider="modal"),
    tools=[custom_bash]  # Overrides auto-generated bash tool
)
```

## Lifecycle

The sandbox provider is created when you call `create_deep_agent()` or `create_sandbox_provider()`, but the actual sandbox container is created **lazily** on first file operation:

```
create_deep_agent(sandbox=SandboxConfig(...))
    ↓
ModalSandboxProvider created (no container yet)
    ↓
FilesystemMiddleware detects execute() → generates bash tool
    ↓
agent.invoke() called
    ↓
First /workspace/ file operation
    ↓
Sandbox container created (lazy initialization)
    ├── Modal Sandbox API called
    ├── Container started
    ├── Sandbox ID assigned
    ├── Bootstrap runs (if configured)
    └── Operation executes
    ↓
All subsequent operations use SAME sandbox container
    ├── Files persist across tool calls
    ├── Installed packages remain available
    └── Working directory state maintained
    ↓
agent.cleanup() OR context manager __exit__
    ↓
provider.cleanup() called (async)
    ↓
Sandbox container terminated, resources freed
```

**Key Points:**
- Provider created immediately, container created lazily
- Container persists for entire agent session
- Cleanup is async but handled automatically
- Context manager ensures cleanup even if errors occur

## Complete Examples

### Example 1: Upload Data Then Analyze

```python
import tempfile
from deepagents import create_deep_agent
from deepagents.backends.sandbox import SandboxConfig, create_sandbox_provider

# Create data file
with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
    f.write("name,age\\nAlice,30\\nBob,25\\n")
    csv_file = f.name

config = SandboxConfig(provider="modal")

# Use context manager for provider
with create_sandbox_provider(config) as provider:
    # Upload data before creating agent
    provider.upload_file(local_path=csv_file, path="/workspace/data.csv")

    # Create agent with same sandbox provider
    with create_deep_agent(
        sandbox=config,
        system_prompt="You are a data analyst. Use /workspace/ for data files."
    ) as agent:
        # Agent can now access uploaded data
        result = agent.invoke({
            "messages": [{
                "role": "user",
                "content": "Read /workspace/data.csv and create a Python script to analyze it"
            }]
        })
        print(result["messages"][-1].content)
# Both provider and agent cleaned up automatically
```

### Example 2: Multi-Turn Conversation

```python
from deepagents import create_deep_agent
from deepagents.backends.sandbox import SandboxConfig

with create_deep_agent(sandbox=SandboxConfig(provider="modal")) as agent:
    # Turn 1
    r1 = agent.invoke({
        "messages": [{
            "role": "user",
            "content": "Create a calculator in /workspace/calc.py"
        }]
    })

    # Turn 2 - same sandbox!
    r2 = agent.invoke({
        "messages": [{
            "role": "user",
            "content": "Add multiply and divide functions"
        }]
    })

    # Turn 3
    r3 = agent.invoke({
        "messages": [{
            "role": "user",
            "content": "Create tests in /workspace/test_calc.py"
        }]
    })
# All turns used same sandbox, cleanup automatic
```

## Cost


**Always call agent.cleanup() or use context manager to avoid unnecessary costs!**

## Best Practices

### 1. Use Context Manager

```python
# ✅ Good - cleanup guaranteed
with create_deep_agent(sandbox=...) as agent:
    agent.invoke(...)

# ⚠️ OK - but must remember cleanup
agent = create_deep_agent(sandbox=...)
agent.invoke(...)
agent.cleanup()  # Don't forget!
```

### 2. Upload Data Before Agent Starts

```python
config = SandboxConfig(provider="modal")

# Use context manager for provider
with create_sandbox_provider(config) as provider:
    # Upload first
    provider.upload_file(local_path="./data.csv", path="/workspace/data.csv")

    # Then create agent with same config
    with create_deep_agent(sandbox=config) as agent:
        agent.invoke(...)
# Both provider and agent cleaned up automatically
```

### 3. Use Bootstrap for Heavy Setup

```python
# ✅ Good - install once during bootstrap
config = SandboxConfig(
    provider="modal",
    bootstrap=BootstrapConfig(
        setup_script="pip install numpy pandas torch"
    )
)

# ❌ Bad - agent reinstalls every session
# "run pip install numpy pandas torch"
```

### 4. Set Reasonable Timeouts

```python
# ✅ Good - match expected session length
config = SandboxConfig(timeout_seconds=1800)  # 30 min

# ❌ Bad - too long
config = SandboxConfig(timeout_seconds=86400)  # 24 hours!
```

## Troubleshooting

### Sandbox timeout

**Issue**: "Sandbox timeout" error

**Solution**: Increase timeout
```python
config = SandboxConfig(timeout_seconds=3600)  # 1 hour
```

### Out of memory

**Issue**: Container runs out of memory

**Solution**: Increase memory
```python
config = SandboxConfig(memory_mb=4096)  # 4GB
```

### Forgot to cleanup

**Issue**: Sandbox still running, costing money

**Solution**: Always use context manager or call cleanup()
```python
# Best practice
with create_deep_agent(sandbox=...) as agent:
    agent.invoke(...)
```

## Advanced: Custom Backend Routing

For custom backend configurations, use the `backend` parameter instead of `sandbox`:

### Pattern 1: Custom Route Paths

```python
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend
from deepagents.backends.sandbox import SandboxConfig, create_sandbox_provider

config = SandboxConfig(provider="modal")

# Use context manager for automatic cleanup
with create_sandbox_provider(config) as provider:
    def create_backend(runtime):
        return CompositeBackend(
            default=StateBackend(runtime),
            routes={
                "/workspace/": provider,
                "/data/": provider,  # Custom routing - multiple paths to sandbox
            }
        )

    agent = create_deep_agent(backend=create_backend)
    agent.invoke({
        "messages": [{
            "role": "user",
            "content": "Save data to /data/file.txt and /workspace/code.py"
        }]
    })
# Cleanup happens automatically here
```

### Pattern 2: Multiple Sandboxes

```python
from deepagents.backends import CompositeBackend, StateBackend
from deepagents.backends.sandbox import SandboxConfig, create_sandbox_provider

# Use nested context managers for multiple providers
with create_sandbox_provider(SandboxConfig(provider="modal")) as modal_provider, \
     create_sandbox_provider(SandboxConfig(provider="daytona")) as daytona_provider:

    def create_backend(runtime):
        return CompositeBackend(
            default=StateBackend(runtime),
            routes={
                "/modal/": modal_provider,
                "/daytona/": daytona_provider,
            }
        )

    agent = create_deep_agent(backend=create_backend)
    agent.invoke({
        "messages": [{
            "role": "user",
            "content": "Save to /modal/file1.txt and /daytona/file2.txt"
        }]
    })
# Both providers cleaned up automatically here
```

**Important Notes:**
- When using `backend` parameter, use context managers for provider cleanup
- Providers implement `BackendProtocol` directly - no wrapper needed
- Context manager `with provider:` automatically handles async cleanup
- The `sandbox` parameter is simpler and recommended for most use cases
