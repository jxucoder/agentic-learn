# agentic-learn

Minimal, extensible ML/DS agent harness inspired by [pi-mono](https://github.com/badlogic/pi-mono).

## Features

- **Minimal core**: Simple agent loop that runs until the task is done
- **Extensible**: Add capabilities via tools and extensions
- **ML/DS focused**: Built-in tools for data science workflows
- **Multi-provider**: Support for Anthropic, OpenAI, and local LLMs

## Installation

```bash
pip install -e .

# With optional dependencies
pip install -e ".[gpu]"      # GPU monitoring
pip install -e ".[data]"     # pandas, datasets
pip install -e ".[experiment]" # mlflow, wandb
pip install -e ".[all]"      # Everything
```

## Quick Start

```bash
# Interactive mode
ds-agent

# Single message
ds-agent "Load the iris dataset and show basic statistics"

# With specific model
ds-agent -m claude-sonnet-4-20250514 "Train a simple classifier"
```

## Architecture

```
agentic_learn/
├── core/
│   ├── agent.py      # Main agent loop
│   ├── types.py      # Core types (Message, ToolCall, etc.)
│   ├── tool.py       # Tool base class and decorator
│   └── extension.py  # Extension system
├── tools/
│   ├── python_tool.py    # Python code execution
│   ├── gpu_tool.py       # GPU monitoring
│   ├── data_tool.py      # Dataset loading/exploration
│   └── experiment_tool.py # Experiment tracking
├── extensions/
│   ├── papers.py     # arXiv/Semantic Scholar search
│   ├── ray_ext.py    # Ray distributed computing
│   └── wandb_ext.py  # Weights & Biases integration
└── cli.py            # CLI entry point
```

## Core Tools

| Tool | Description |
|------|-------------|
| `python` | Execute Python code with persistent namespace |
| `gpu` | Monitor GPU status, memory, processes |
| `data` | Load and explore datasets (CSV, Parquet, HuggingFace) |
| `experiment` | Track experiments, log metrics |

## Extensions

Extensions add optional capabilities:

| Extension | Tools Added |
|-----------|-------------|
| `papers` | `papers_search`, `papers_read` - Search arXiv/Semantic Scholar |
| `ray` | `ray_cluster`, `ray_run` - Distributed computing |
| `wandb` | `wandb` - Weights & Biases tracking |

### Loading Extensions

```python
from agentic_learn import Agent, AgentConfig
from agentic_learn.extensions import PapersExtension

agent = Agent(config=AgentConfig())
agent.load_extension(PapersExtension())
```

### Creating Custom Extensions

```python
from agentic_learn.core import Extension, ExtensionAPI, Tool, ToolResult

class MyExtension(Extension):
    name = "my-extension"

    def setup(self, api: ExtensionAPI):
        api.register_tool(MyCustomTool())
        api.on(EventType.TOOL_CALL_START, self.on_tool_call)
```

## Agent Loop Design

Inspired by pi-mono's approach:

1. **No max steps** - Loop continues until the agent decides it's done
2. **Tool validation** - Arguments validated before execution
3. **Event-driven** - Subscribe to events for monitoring/logging
4. **Extensible** - Add tools and hooks without modifying core

```python
async for event in agent.run("Analyze the dataset"):
    if event.type == EventType.MESSAGE_DELTA:
        print(event.data["text"], end="")
    elif event.type == EventType.TOOL_CALL_START:
        print(f"Using tool: {event.data['tool']}")
```

## Environment Variables

- `ANTHROPIC_API_KEY` - Anthropic API key
- `OPENAI_API_KEY` - OpenAI API key
- `WANDB_API_KEY` - Weights & Biases API key

## License

MIT
