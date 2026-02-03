# agentic-learn

A **coding agent for data science** inspired by [pi-mono](https://github.com/badlogic/pi-mono).

Write code, run experiments, train models - all through natural language.

## Install

```bash
pip install -e .
```

## Usage

```bash
# Interactive mode (default: Tier 2 tools)
ds-agent

# Single command
ds-agent "Create a CNN for MNIST and train it"

# Choose tool tier
ds-agent -t 1  # Core only: read, write, edit, bash, python
ds-agent -t 2  # + DS essentials: gpu, data, experiment (default)
ds-agent -t 3  # + Advanced: jobs, tune, viz, notebook, repro
```

## Tool Tiers

Tools are organized in tiers. Load what you need.

### Tier 1: Core (5 tools)
The fundamentals every coding agent needs.

| Tool | Description |
|------|-------------|
| `read` | Read files and directories |
| `write` | Create and write files |
| `edit` | Edit files (find-replace) |
| `bash` | Run shell commands |
| `python` | Execute Python code |

### Tier 2: DS Essentials (3 tools)
Essential for ML/DS work. Reliable, structured output.

| Tool | Description |
|------|-------------|
| `gpu` | Monitor GPU resources |
| `data` | Load and explore datasets |
| `experiment` | Track experiments |

### Tier 3: Advanced (5 tools)
For complex, long-running workflows.

| Tool | Description |
|------|-------------|
| `jobs` | Background jobs with checkpointing |
| `tune` | Hyperparameter tuning |
| `viz` | Create visualizations |
| `notebook` | Jupyter notebooks |
| `repro` | Reproducibility (seeds, snapshots) |

## Why Tiers?

**Tier 1** - Minimal. The agent can do anything with these 5 tools.

**Tier 2** - But dedicated tools are more reliable. `gpu action="status"` is better than writing nvidia-smi parsing code every time.

**Tier 3** - For advanced workflows. Long training jobs, hyperparameter sweeps, reproducibility.

## Extensions

Add more capabilities via extensions:

```python
from agentic_learn import Agent, AgentConfig
from agentic_learn.extensions import PapersExtension, RayExtension

agent = Agent(config=AgentConfig())
agent.load_extension(PapersExtension())  # arXiv/paper search
agent.load_extension(RayExtension())      # Distributed computing
```

Built-in extensions:
- `papers` - Search arXiv, Semantic Scholar
- `ray` - Ray distributed computing
- `wandb` - Weights & Biases tracking

## Architecture

```
agentic_learn/
├── core/
│   ├── agent.py      # Agent loop (runs until done)
│   ├── types.py      # Message, ToolCall, etc.
│   ├── tool.py       # Tool base class
│   └── extension.py  # Extension system
├── tools/            # 13 built-in tools
├── extensions/       # Optional extensions
└── cli.py            # ds-agent command
```

## Design

Inspired by pi-mono:

1. **No max steps** - Runs until the agent decides it's done
2. **Minimal core** - 5 tools can do anything
3. **Tiered complexity** - Add tools as needed
4. **Extensible** - Extensions for specialized tasks

## API

```python
from agentic_learn import Agent, AgentConfig
from agentic_learn.tools import get_tools

agent = Agent(config=AgentConfig())

# Register tier 2 tools (default)
for tool in get_tools(tier=2):
    agent.register_tool(tool)

# Run
async for event in agent.run("Analyze data.csv"):
    print(event)
```

## Environment

```bash
export ANTHROPIC_API_KEY=...  # or OPENAI_API_KEY
```

## License

MIT
