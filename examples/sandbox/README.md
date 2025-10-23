# Modal Sandbox Example

This example demonstrates using DeepAgents with Modal sandboxes for secure shell execution.

## Features

- ✅ OpenAI GPT-5 Mini integration
- ✅ Secure command execution in Modal sandboxes
- ✅ Pre-installed packages (pytest, black, ruff, git)
- ✅ Automatic cleanup on exit
- ✅ LangGraph Studio compatible

## Quick Start

### Interactive TUI (Two Options)

We provide two beautiful terminal UI experiences:

#### Option 1: Minimalist Chat (Recommended)

Clean, snappy prompt-toolkit interface:

```bash
python examples/sandbox/chat_promptkit_tui.py
```

**Features:**
- ⚡ Lightning fast & minimal
- 🎨 Emerald green theme with clean "DEEP AGENTS" ASCII art header
- 💬 Streaming word-by-word responses
- ⣾ Animated dots spinner while agent is thinking
- 🔧 Tool calls shown inline (shell, read_file, write_file, etc.)
- 📁 Local file sync to sandbox (see `possibly_unsafe_code.py` demo)
- Simple `>` prompt (no clutter, no borders)

#### Option 2: Full TUI (Textual)

Rich widget-based interface with more features:

```bash
python examples/sandbox/chat_textual_tui.py
```

**Features:**
- 📦 Structured message blocks with borders
- ⌨️  Keyboard shortcuts (Ctrl+C quit, Ctrl+L clear)
- 🎨 Emerald green theme throughout
- 💬 Streaming responses
- 🔧 Tool calls with icons

**Which to choose?**
- **chat_promptkit_tui.py** - Faster startup, cleaner feel, better for quick coding sessions
- **chat_textual_tui.py** - More structured, better for complex multi-turn conversations

---

## Setup

### 1. Install Dependencies

```bash
# From project root
uv sync

# Or if you want to install just for this example
cd examples/sandbox
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file in this directory:

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

Your `.env` should look like:
```bash
OPENAI_API_KEY=sk-your-actual-openai-key-here
```

### 3. Setup Modal

```bash
# One-time setup
modal setup
```

This will:
- Open your browser to log into Modal
- Create/authenticate your Modal account
- Save credentials to `~/.modal.toml`

## Usage

### Quick Test

```bash
python main.py
```

This will:
1. Create a Modal sandbox with Python 3.11 + pytest/black/ruff
2. Ask the agent to run `pytest --version`
3. Display the result
4. Automatically clean up the sandbox

### Advanced Examples

Run the full demo with multiple examples:

```bash
python demo.py
```

This includes:
- Basic package installation
- File synchronization
- Custom images
- Interactive sessions

### LangGraph Studio

Open this example in LangGraph Studio for visual debugging:

```bash
# From this directory
langgraph dev
```

Then open http://localhost:8123 in your browser.

## How It Works

### Agent Configuration

The agent ([sandbox_agent.py](sandbox_agent.py:13-27)) uses explicit middleware instantiation:

```python
from deepagents.middleware.sandbox_shell import SandboxShellMiddleware

sandbox_middleware = SandboxShellMiddleware(
    pip_packages=["pytest", "black", "ruff"],
    apt_packages=["git"],
    timeout=3600,
    verbose=True,
)

agent = create_deep_agent(
    model=ChatOpenAI(model="gpt-4o-mini-2025-08-07"),
    middleware=[sandbox_middleware],
)
```

Alternatively, you can use the convenience parameter:
```python
agent = create_deep_agent(
    model=ChatOpenAI(model="gpt-4o-mini-2025-08-07"),
    use_sandbox_shell=True,
    sandbox_config={
        "pip_packages": ["pytest", "black", "ruff"],
        "apt_packages": ["git"],
        "timeout": 3600,
        "verbose": True,
    }
)
```

### Sandbox Behavior

When you invoke the agent:
1. **First command:** Creates Modal sandbox (~5-30s for image build)
2. **Subsequent commands:** Reuses same sandbox (<1s)
3. **On exit:** Automatically terminates sandbox

### Security

- Commands run in isolated gVisor containers
- No access to your local filesystem (unless synced)
- No access to Modal infrastructure
- Network access configurable

## Switching to Anthropic Claude

To use Claude instead of OpenAI:

1. Update `.env`:
```bash
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

2. Edit [sandbox_agent.py](sandbox_agent.py):
```python
from deepagents.middleware.sandbox_shell import SandboxShellMiddleware

sandbox_middleware = SandboxShellMiddleware(
    pip_packages=["pytest", "black", "ruff"],
    apt_packages=["git"],
    timeout=3600,
)

# No model parameter = uses Claude Sonnet 4.5
agent = create_deep_agent(
    middleware=[sandbox_middleware],
)
```

## Configuration Options

### Sandbox Config

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `pip_packages` | list[str] | [] | Python packages to install |
| `apt_packages` | list[str] | [] | System packages to install |
| `custom_image` | modal.Image | None | Custom Modal image (advanced) |
| `image_config` | dict | None | Declarative image config (see below) |
| `sync_cwd` | bool | False | Sync current directory to sandbox |
| `sync_paths` | list[str] | [] | Specific paths to sync |
| `sync_exclude` | list[str] | [".git", ...] | Patterns to exclude |
| `timeout` | int | 3600 | Sandbox timeout (seconds) |
| `verbose` | bool | True | Show Rich notifications |

### Example: Declarative Image Config

Instead of passing `custom_image` (which requires importing `modal`), use `image_config` for declarative setup:

```python
from deepagents.middleware.sandbox_shell import SandboxShellMiddleware

sandbox_middleware = SandboxShellMiddleware(
    image_config={
        "base": "python:3.11-slim",
        "pip_packages": ["torch", "transformers"],
        "apt_packages": ["ffmpeg", "git"],
        "commands": [
            "curl -O https://example.com/model.bin",
            "chmod +x /usr/local/bin/custom-tool"
        ]
    }
)

agent = create_deep_agent(
    model=ChatOpenAI(model="gpt-4o-mini-2025-08-07"),
    middleware=[sandbox_middleware],
)
```

### Example: Local Filesystem + Sandbox Shell

Want to read/write real local files but execute commands in a secure sandbox? Use `use_sandbox_shell_with_local_fs`:

```python
agent = create_deep_agent(
    model=ChatOpenAI(model="gpt-4o-mini-2025-08-07"),
    use_sandbox_shell_with_local_fs=True,
    sandbox_config={
        "pip_packages": ["pytest", "black"],
        "apt_packages": ["git"],
    }
)
```

This combination allows the agent to:
- Read and write actual files on your local disk
- Execute shell commands in an isolated Modal sandbox
- Useful for: "Edit my local Python files, but run tests in a clean environment"

### Example: Sync Local Files

```python
sandbox_config={
    "sync_cwd": True,  # Sync entire current directory
    "sync_exclude": [".git", "*.pyc", "__pycache__", "venv"],
    "pip_packages": ["pytest"],
}
```

Then the agent can access your files at `/workspace/` in the sandbox.

## Troubleshooting

### "Modal token not found"
Run `modal setup` to authenticate.

### "OPENAI_API_KEY not set"
Create `.env` file with your API key.

### "Module 'modal' not found"
Run `uv sync` or `pip install -r requirements.txt`.

### "Module 'textual' not found" (chat_tui.py only)
Run `pip install textual>=0.89.0` or `uv sync`.

### "Module 'prompt_toolkit' not found" (chat_prompt.py only)
Run `pip install prompt-toolkit>=3.0.47` or `uv sync`.

### Sandbox creation slow
First run builds the image (~30s). Subsequent runs are fast (<5s).

### Command timeout
Increase timeout in sandbox_config: `"timeout": 7200` (2 hours).

## Files

- **[chat_prompt.py](chat_prompt.py)** - Minimalist prompt-toolkit TUI (recommended)
- **[chat_tui.py](chat_tui.py)** - Full-featured Textual TUI
- **[chat_agent.py](chat_agent.py)** - Shared agent factory for TUIs
- [sandbox_agent.py](sandbox_agent.py) - Explicit middleware example
- [main.py](main.py) - Minimal one-shot test script
- [demo.py](demo.py) - Comprehensive examples
- [langgraph.json](langgraph.json) - LangGraph Studio config
- [requirements.txt](requirements.txt) - Python dependencies

## Learn More

- [Modal Sandboxes Documentation](https://modal.com/docs/guide/sandbox)
- [DeepAgents Documentation](../../README.md)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
