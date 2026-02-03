# agentic-learn

Minimal, extensible ML/DS agent harness inspired by [pi-mono](https://github.com/badlogic/pi-mono).

## Features

- **Minimal core**: Simple agent loop that runs until the task is done
- **Extensible**: Add capabilities via tools and extensions
- **ML/DS focused**: Built-in tools for data science workflows
- **Long-running tasks**: Background jobs with progress tracking and checkpointing
- **Reproducibility**: Seed management, environment snapshots, config tracking
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
│   ├── experiment_tool.py # Experiment tracking
│   ├── jobs_tool.py      # Background job management
│   ├── notebook_tool.py  # Jupyter notebook manipulation
│   ├── tuning_tool.py    # Hyperparameter tuning
│   ├── viz_tool.py       # Visualization creation
│   └── repro_tool.py     # Reproducibility management
├── extensions/
│   ├── papers.py     # arXiv/Semantic Scholar search
│   ├── ray_ext.py    # Ray distributed computing
│   └── wandb_ext.py  # Weights & Biases integration
└── cli.py            # CLI entry point
```

## Tools

### Core Tools

| Tool | Description |
|------|-------------|
| `python` | Execute Python code with persistent namespace |
| `gpu` | Monitor GPU status, memory, processes |
| `data` | Load and explore datasets (CSV, Parquet, HuggingFace) |
| `experiment` | Track experiments, log metrics |

### DS-Specific Tools

| Tool | Description |
|------|-------------|
| `jobs` | Background jobs with progress tracking and checkpointing |
| `notebook` | Create and manipulate Jupyter notebooks |
| `tune` | Hyperparameter tuning with search strategies |
| `viz` | Create visualizations (line, scatter, heatmap, confusion matrix, etc.) |
| `repro` | Reproducibility: seeds, snapshots, configs, hashes |

## Long-Running Tasks

The `jobs` tool handles tasks that take minutes to hours:

```python
# Submit a training job to run in background
jobs action="submit" name="resnet-training" code="""
for epoch in range(100):
    loss = train_epoch(model, data)
    progress.update(current=epoch, total=100, loss=loss)
    if epoch % 10 == 0:
        progress.checkpoint({'model': model.state_dict(), 'epoch': epoch})
"""

# Check progress
jobs action="status" job_id="abc123"

# Resume from checkpoint if interrupted
jobs action="resume" job_id="abc123"
```

Features:
- Progress tracking with metrics
- Automatic checkpointing
- Resume from failure
- Cancellation support

## Hyperparameter Tuning

```python
# Create a tuning study
tune action="create" study="lr-search" space='[
  {"name": "lr", "type": "log_uniform", "low": 1e-5, "high": 1e-1},
  {"name": "batch_size", "type": "choice", "choices": [16, 32, 64]}
]' direction="minimize"

# Get suggested parameters
tune action="suggest" study="lr-search"

# Report results
tune action="report" study="lr-search" trial_id=0 metric=0.85

# Get best parameters
tune action="best" study="lr-search"
```

## Reproducibility

```python
# Set seeds for all frameworks
repro action="seed" value=42

# Capture environment state
repro action="snapshot" name="baseline"

# Save experiment config
repro action="config" action="save" name="exp1" data='{"lr": 0.001}'

# Hash model/data files
repro action="hash" path="model.pt"
```

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

## Example Workflow

```python
# 1. Set up reproducibility
repro action="seed" value=42
repro action="snapshot" name="baseline"

# 2. Load and explore data
data action="load" source="train.csv"
data action="profile" source="train.csv"

# 3. Create analysis notebook
notebook action="create" path="analysis.ipynb" title="Model Training"

# 4. Start hyperparameter search
tune action="create" study="model-search" space='[...]'

# 5. Submit training job
jobs action="submit" name="training" code="..."

# 6. Monitor progress
jobs action="status" job_id="..."
gpu action="status"

# 7. Visualize results
viz action="plot" type="learning_curve" data='{"train": [...], "val": [...]}'

# 8. Get best model
tune action="best" study="model-search"
```

## Environment Variables

- `ANTHROPIC_API_KEY` - Anthropic API key
- `OPENAI_API_KEY` - OpenAI API key
- `WANDB_API_KEY` - Weights & Biases API key

## License

MIT
