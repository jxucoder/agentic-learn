# agentic-learn

A self-evolving ML agent that iteratively writes, runs, and improves
scikit-learn solutions for tabular data tasks.

## How it works

```
              Experiment Journal
              (sorted by metric)
                     │
                     ▼
    ┌─────────────────────────────┐
    │         Briefing            │  Build task + journal into a prompt
    └──────────────┬──────────────┘
                   │
                   ▼
    ┌─────────────────────────────┐
    │       Coding Agent          │  codex exec --full-auto
    │                             │  Reads data, writes code, runs it,
    │                             │  debugs errors, reports metric
    └──────────────┬──────────────┘
                   │
                   ▼
    ┌─────────────────────────────┐
    │         Record              │  Append to journal.jsonl
    └──────────────┬──────────────┘
                   │
                   └──────► next step
```

Each step, a coding agent receives a task briefing that includes the
**full experiment history sorted by score**. The agent explores the data,
writes a scikit-learn solution, runs it, debugs any errors, and reports
the metric. The framework records the result and moves on.

The agent handles Think-Code-Verify internally — there is no separate
code generation, execution, or error-retry logic in the framework.
The framework's only job is to maintain the journal and brief the agent.

## Design

Three files.

| File | What it does |
|---|---|
| **journal.py** | Append-only experiment log. Each entry stores code, hypothesis, and metric. Persisted as JSON lines. Sorted by metric for context. |
| **agent.py** | Invokes `codex exec --full-auto`. The agent writes `solution.py` and `result.json` in a working directory. |
| **loop.py** | Builds the briefing, calls the agent, records the result. The entire loop is a `for` over `max_steps`. |

### Why this shape

**The coding agent is the execution unit.** It can read files, write code,
run it, see errors, fix them, and iterate — all within a single invocation.
Wrapping an LLM with manual code extraction, subprocess execution, and
linter retries is reimplementing what the agent already does, but worse.

**A flat journal, not a tree or graph.** Every experiment is independent —
informed by all past results, but not derived from a specific parent.
The full sorted history gives the agent everything it needs to avoid
repeating past work and to build on what succeeded.

**Explicit diversity instruction.** Without it, the agent collapses into
incremental hyperparameter tweaks. The briefing instructs the agent to
"try something meaningfully different from what's in the journal."

**Steering via instructions.** `TaskConfig.instructions` lets a human
redirect the agent mid-run ("focus on ensemble methods", "try feature
selection") without changing the code.

## Quickstart

Requires [codex CLI](https://github.com/openai/codex) installed and
authenticated.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

```python
from agentic_learn import TaskConfig, evolve

task = TaskConfig(
    description="Binary classification: predict survival on the Titanic.",
    data_path="/path/to/titanic.csv",
    target_column="Survived",
    metric="f1_score",
)

best = evolve(task, model="o4-mini", max_steps=10)
print(best.metric_value)
```

The best solution is saved to `./output/best_solution.py`. The full
experiment history is in `./output/journal.jsonl`. Each step's working
directory (with the agent's `solution.py` and `result.json`) is preserved
under `./output/step_000/`, `./output/step_001/`, etc.

### Configuration

| Parameter | Default | What it controls |
|---|---|---|
| `model` | `o4-mini` | Model passed to `codex exec -m` |
| `max_steps` | `10` | Number of agent invocations |
| `timeout` | `300` | Seconds before an agent run is killed |
| `output_dir` | `./output` | Where journal and step directories are written |
| `task.instructions` | `""` | Optional human steering (free text) |

### Benchmarks

| Benchmark | Task | Baseline | Good | Command |
|---|---|---|---|---|
| **Titanic** | Binary classification (F1) | ~0.62 | ~0.82+ | `python examples/titanic.py` |
| **California Housing** | Regression (R²) | ~0.55 | ~0.85+ | `python examples/california_housing.py` |

See [`examples/`](examples/) for details on each benchmark.

## Project structure

```
src/agentic_learn/
    __init__.py     Exports: TaskConfig, evolve, Journal, Experiment
    journal.py      Experiment dataclass + append-only Journal
    agent.py        codex exec wrapper
    loop.py         Briefing + evolve loop
examples/
    titanic.py                  Binary classification benchmark
    california_housing.py       Regression benchmark
```

## Requirements

- Python >= 3.11
- [codex CLI](https://github.com/openai/codex) installed and authenticated
