"""Experiment tracking tool."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from agentic_learn.core.tool import Tool, ToolContext, ToolParameter
from agentic_learn.core.types import ToolResult


@dataclass
class Run:
    """A single experiment run."""

    id: str
    name: str
    experiment: str
    status: str = "running"  # running, completed, failed
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, list[tuple[int, float]]] = field(default_factory=dict)  # step -> value history
    tags: dict[str, str] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """Log a metric value."""
        if key not in self.metrics:
            self.metrics[key] = []
        if step is None:
            step = len(self.metrics[key])
        self.metrics[key].append((step, value))

    def get_latest_metrics(self) -> dict[str, float]:
        """Get the latest value for each metric."""
        return {k: v[-1][1] for k, v in self.metrics.items() if v}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "experiment": self.experiment,
            "status": self.status,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "params": self.params,
            "metrics": self.metrics,
            "tags": self.tags,
            "artifacts": self.artifacts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Run:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            experiment=data["experiment"],
            status=data.get("status", "running"),
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            params=data.get("params", {}),
            metrics=data.get("metrics", {}),
            tags=data.get("tags", {}),
            artifacts=data.get("artifacts", []),
        )


class ExperimentTool(Tool):
    """Track ML experiments, metrics, and artifacts.

    A lightweight built-in experiment tracker. Can also integrate
    with MLflow or Weights & Biases via extensions.
    """

    name = "experiment"
    description = """Track experiments, log metrics, and manage runs.

Actions:
- start: Start a new experiment run
- end: End the current run (status: completed or failed)
- log: Log metrics (key=value pairs)
- param: Log parameters/hyperparameters
- tag: Add tags to current run
- artifact: Register an artifact (file path)
- status: Show current run status
- list: List all runs in an experiment
- compare: Compare metrics across runs
- best: Get the best run by a metric
- delete: Delete a run

The tracker stores data locally in .ds-agent/experiments/.
Use extensions for MLflow/W&B integration.

Examples:
- start experiment="mnist" name="cnn-v1"
- param learning_rate=0.001 batch_size=32
- log accuracy=0.95 loss=0.12 step=100
- end status="completed"
- compare experiment="mnist" metric="accuracy\""""

    parameters = [
        ToolParameter(
            name="action",
            type=str,
            description="Action: start, end, log, param, tag, artifact, status, list, compare, best, delete",
            required=True,
        ),
        ToolParameter(
            name="options",
            type=dict,
            description="Action-specific options (experiment, name, metrics, params, etc.)",
            required=False,
            default={},
        ),
    ]

    def __init__(self, storage_dir: str = ".ds-agent/experiments"):
        super().__init__()
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.current_run: Run | None = None

    def _generate_run_id(self) -> str:
        """Generate a unique run ID."""
        import hashlib
        timestamp = str(time.time()).encode()
        return hashlib.sha256(timestamp).hexdigest()[:12]

    def _save_run(self, run: Run) -> None:
        """Save a run to disk."""
        exp_dir = self.storage_dir / run.experiment
        exp_dir.mkdir(parents=True, exist_ok=True)

        run_file = exp_dir / f"{run.id}.json"
        with open(run_file, "w") as f:
            json.dump(run.to_dict(), f, indent=2)

    def _load_run(self, experiment: str, run_id: str) -> Run | None:
        """Load a run from disk."""
        run_file = self.storage_dir / experiment / f"{run_id}.json"
        if not run_file.exists():
            return None

        with open(run_file) as f:
            return Run.from_dict(json.load(f))

    def _list_runs(self, experiment: str) -> list[Run]:
        """List all runs in an experiment."""
        exp_dir = self.storage_dir / experiment
        if not exp_dir.exists():
            return []

        runs = []
        for run_file in exp_dir.glob("*.json"):
            with open(run_file) as f:
                runs.append(Run.from_dict(json.load(f)))

        return sorted(runs, key=lambda r: r.start_time, reverse=True)

    async def execute(
        self,
        ctx: ToolContext,
        action: str,
        options: dict[str, Any] | None = None,
    ) -> ToolResult:
        """Execute experiment tracking action."""
        options = options or {}
        action = action.lower()

        try:
            if action == "start":
                return await self._start(options)
            elif action == "end":
                return await self._end(options)
            elif action == "log":
                return await self._log(options)
            elif action == "param":
                return await self._param(options)
            elif action == "tag":
                return await self._tag(options)
            elif action == "artifact":
                return await self._artifact(options)
            elif action == "status":
                return await self._status(options)
            elif action == "list":
                return await self._list(options)
            elif action == "compare":
                return await self._compare(options)
            elif action == "best":
                return await self._best(options)
            elif action == "delete":
                return await self._delete(options)
            else:
                return ToolResult(
                    tool_call_id="",
                    content=f"Unknown action: {action}",
                    is_error=True,
                )
        except Exception as e:
            return ToolResult(
                tool_call_id="",
                content=f"Error: {str(e)}",
                is_error=True,
            )

    async def _start(self, options: dict[str, Any]) -> ToolResult:
        """Start a new experiment run."""
        experiment = options.get("experiment", "default")
        name = options.get("name", f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}")

        run_id = self._generate_run_id()
        self.current_run = Run(
            id=run_id,
            name=name,
            experiment=experiment,
            params=options.get("params", {}),
            tags=options.get("tags", {}),
        )

        self._save_run(self.current_run)

        return ToolResult(
            tool_call_id="",
            content=f"""Experiment run started:
  Experiment: {experiment}
  Run Name: {name}
  Run ID: {run_id}
  Started: {self.current_run.start_time.isoformat()}

Use 'log' to record metrics, 'param' for hyperparameters.""",
            metadata={"run_id": run_id, "experiment": experiment},
        )

    async def _end(self, options: dict[str, Any]) -> ToolResult:
        """End the current run."""
        if self.current_run is None:
            return ToolResult(
                tool_call_id="",
                content="No active run to end.",
                is_error=True,
            )

        status = options.get("status", "completed")
        self.current_run.status = status
        self.current_run.end_time = datetime.now()

        self._save_run(self.current_run)

        duration = self.current_run.end_time - self.current_run.start_time
        metrics = self.current_run.get_latest_metrics()

        result = f"""Run ended:
  Run ID: {self.current_run.id}
  Status: {status}
  Duration: {duration}
  Final Metrics: {json.dumps(metrics, indent=2)}"""

        self.current_run = None

        return ToolResult(tool_call_id="", content=result)

    async def _log(self, options: dict[str, Any]) -> ToolResult:
        """Log metrics."""
        if self.current_run is None:
            return ToolResult(
                tool_call_id="",
                content="No active run. Start a run first with 'start'.",
                is_error=True,
            )

        step = options.pop("step", None)
        logged = []

        for key, value in options.items():
            if isinstance(value, (int, float)):
                self.current_run.log_metric(key, float(value), step)
                logged.append(f"{key}={value}")

        self._save_run(self.current_run)

        return ToolResult(
            tool_call_id="",
            content=f"Logged metrics: {', '.join(logged)}" + (f" (step={step})" if step else ""),
        )

    async def _param(self, options: dict[str, Any]) -> ToolResult:
        """Log parameters."""
        if self.current_run is None:
            return ToolResult(
                tool_call_id="",
                content="No active run. Start a run first with 'start'.",
                is_error=True,
            )

        self.current_run.params.update(options)
        self._save_run(self.current_run)

        return ToolResult(
            tool_call_id="",
            content=f"Logged parameters: {json.dumps(options)}",
        )

    async def _tag(self, options: dict[str, Any]) -> ToolResult:
        """Add tags to current run."""
        if self.current_run is None:
            return ToolResult(
                tool_call_id="",
                content="No active run. Start a run first with 'start'.",
                is_error=True,
            )

        for key, value in options.items():
            self.current_run.tags[key] = str(value)

        self._save_run(self.current_run)

        return ToolResult(
            tool_call_id="",
            content=f"Added tags: {json.dumps(options)}",
        )

    async def _artifact(self, options: dict[str, Any]) -> ToolResult:
        """Register an artifact."""
        if self.current_run is None:
            return ToolResult(
                tool_call_id="",
                content="No active run. Start a run first with 'start'.",
                is_error=True,
            )

        path = options.get("path")
        if not path:
            return ToolResult(
                tool_call_id="",
                content="Artifact path required.",
                is_error=True,
            )

        self.current_run.artifacts.append(str(path))
        self._save_run(self.current_run)

        return ToolResult(
            tool_call_id="",
            content=f"Registered artifact: {path}",
        )

    async def _status(self, options: dict[str, Any]) -> ToolResult:
        """Show current run status."""
        if self.current_run is None:
            return ToolResult(
                tool_call_id="",
                content="No active run.",
            )

        run = self.current_run
        duration = datetime.now() - run.start_time
        metrics = run.get_latest_metrics()

        return ToolResult(
            tool_call_id="",
            content=f"""Current Run Status:
  Experiment: {run.experiment}
  Name: {run.name}
  ID: {run.id}
  Status: {run.status}
  Duration: {duration}
  Parameters: {json.dumps(run.params, indent=2)}
  Latest Metrics: {json.dumps(metrics, indent=2)}
  Tags: {json.dumps(run.tags, indent=2)}
  Artifacts: {run.artifacts}""",
        )

    async def _list(self, options: dict[str, Any]) -> ToolResult:
        """List all runs in an experiment."""
        experiment = options.get("experiment", "default")
        runs = self._list_runs(experiment)

        if not runs:
            return ToolResult(
                tool_call_id="",
                content=f"No runs found in experiment '{experiment}'.",
            )

        lines = [f"Runs in '{experiment}':", "=" * 60, ""]
        lines.append(f"{'ID':<14} {'Name':<25} {'Status':<12} {'Date'}")
        lines.append("-" * 60)

        for run in runs:
            date_str = run.start_time.strftime("%Y-%m-%d %H:%M")
            lines.append(f"{run.id:<14} {run.name[:24]:<25} {run.status:<12} {date_str}")

        return ToolResult(
            tool_call_id="",
            content="\n".join(lines),
        )

    async def _compare(self, options: dict[str, Any]) -> ToolResult:
        """Compare metrics across runs."""
        experiment = options.get("experiment", "default")
        metric = options.get("metric")
        runs = self._list_runs(experiment)

        if not runs:
            return ToolResult(
                tool_call_id="",
                content=f"No runs found in experiment '{experiment}'.",
            )

        lines = [f"Comparison for '{experiment}':", "=" * 70, ""]

        # Get all metric names if not specified
        if metric:
            metrics_to_show = [metric]
        else:
            metrics_to_show = set()
            for run in runs:
                metrics_to_show.update(run.metrics.keys())
            metrics_to_show = sorted(metrics_to_show)[:5]  # Limit to 5

        # Build header
        header = f"{'Run':<20}"
        for m in metrics_to_show:
            header += f" {m[:12]:>12}"
        lines.append(header)
        lines.append("-" * 70)

        # Build rows
        for run in runs[:10]:  # Limit to 10 runs
            row = f"{run.name[:19]:<20}"
            latest = run.get_latest_metrics()
            for m in metrics_to_show:
                val = latest.get(m)
                if val is not None:
                    row += f" {val:>12.4f}"
                else:
                    row += f" {'N/A':>12}"
            lines.append(row)

        return ToolResult(
            tool_call_id="",
            content="\n".join(lines),
        )

    async def _best(self, options: dict[str, Any]) -> ToolResult:
        """Get the best run by a metric."""
        experiment = options.get("experiment", "default")
        metric = options.get("metric")
        minimize = options.get("minimize", False)

        if not metric:
            return ToolResult(
                tool_call_id="",
                content="Metric name required.",
                is_error=True,
            )

        runs = self._list_runs(experiment)

        if not runs:
            return ToolResult(
                tool_call_id="",
                content=f"No runs found in experiment '{experiment}'.",
            )

        # Find best run
        best_run = None
        best_value = None

        for run in runs:
            latest = run.get_latest_metrics()
            if metric in latest:
                value = latest[metric]
                if best_value is None:
                    best_value = value
                    best_run = run
                elif minimize and value < best_value:
                    best_value = value
                    best_run = run
                elif not minimize and value > best_value:
                    best_value = value
                    best_run = run

        if best_run is None:
            return ToolResult(
                tool_call_id="",
                content=f"No runs with metric '{metric}' found.",
            )

        return ToolResult(
            tool_call_id="",
            content=f"""Best run by {metric} ({'min' if minimize else 'max'}):
  Run: {best_run.name}
  ID: {best_run.id}
  {metric}: {best_value}
  All metrics: {json.dumps(best_run.get_latest_metrics(), indent=2)}
  Parameters: {json.dumps(best_run.params, indent=2)}""",
        )

    async def _delete(self, options: dict[str, Any]) -> ToolResult:
        """Delete a run."""
        experiment = options.get("experiment", "default")
        run_id = options.get("run_id")

        if not run_id:
            return ToolResult(
                tool_call_id="",
                content="Run ID required.",
                is_error=True,
            )

        run_file = self.storage_dir / experiment / f"{run_id}.json"

        if not run_file.exists():
            return ToolResult(
                tool_call_id="",
                content=f"Run '{run_id}' not found.",
                is_error=True,
            )

        run_file.unlink()

        return ToolResult(
            tool_call_id="",
            content=f"Deleted run '{run_id}' from experiment '{experiment}'.",
        )
