"""Hyperparameter tuning tool."""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from agentic_learn.core.tool import Tool, ToolContext, ToolParameter
from agentic_learn.core.types import ToolResult


@dataclass
class HyperparameterSpace:
    """Definition of hyperparameter search space."""

    name: str
    type: str  # "uniform", "log_uniform", "choice", "int"
    low: float | None = None
    high: float | None = None
    choices: list[Any] | None = None

    def sample(self) -> Any:
        """Sample a value from the space."""
        if self.type == "uniform":
            return random.uniform(self.low, self.high)
        elif self.type == "log_uniform":
            import math
            log_low = math.log(self.low)
            log_high = math.log(self.high)
            return math.exp(random.uniform(log_low, log_high))
        elif self.type == "int":
            return random.randint(int(self.low), int(self.high))
        elif self.type == "choice":
            return random.choice(self.choices)
        else:
            raise ValueError(f"Unknown type: {self.type}")


@dataclass
class Trial:
    """A single hyperparameter trial."""

    id: int
    params: dict[str, Any]
    metric: float | None = None
    status: str = "pending"  # pending, running, completed, failed
    start_time: datetime | None = None
    end_time: datetime | None = None
    error: str | None = None

    @property
    def duration(self) -> float | None:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


@dataclass
class Study:
    """A hyperparameter tuning study."""

    name: str
    space: list[HyperparameterSpace]
    metric_name: str = "metric"
    direction: str = "minimize"  # minimize or maximize
    trials: list[Trial] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def best_trial(self) -> Trial | None:
        completed = [t for t in self.trials if t.status == "completed" and t.metric is not None]
        if not completed:
            return None
        if self.direction == "minimize":
            return min(completed, key=lambda t: t.metric)
        else:
            return max(completed, key=lambda t: t.metric)

    def suggest(self) -> dict[str, Any]:
        """Suggest next hyperparameters to try."""
        return {hp.name: hp.sample() for hp in self.space}

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "space": [
                {
                    "name": hp.name,
                    "type": hp.type,
                    "low": hp.low,
                    "high": hp.high,
                    "choices": hp.choices,
                }
                for hp in self.space
            ],
            "metric_name": self.metric_name,
            "direction": self.direction,
            "trials": [
                {
                    "id": t.id,
                    "params": t.params,
                    "metric": t.metric,
                    "status": t.status,
                    "start_time": t.start_time.isoformat() if t.start_time else None,
                    "end_time": t.end_time.isoformat() if t.end_time else None,
                    "error": t.error,
                }
                for t in self.trials
            ],
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Study:
        space = [
            HyperparameterSpace(
                name=hp["name"],
                type=hp["type"],
                low=hp.get("low"),
                high=hp.get("high"),
                choices=hp.get("choices"),
            )
            for hp in data["space"]
        ]

        study = cls(
            name=data["name"],
            space=space,
            metric_name=data.get("metric_name", "metric"),
            direction=data.get("direction", "minimize"),
            created_at=datetime.fromisoformat(data["created_at"]),
        )

        for t in data.get("trials", []):
            trial = Trial(
                id=t["id"],
                params=t["params"],
                metric=t.get("metric"),
                status=t.get("status", "pending"),
                start_time=datetime.fromisoformat(t["start_time"]) if t.get("start_time") else None,
                end_time=datetime.fromisoformat(t["end_time"]) if t.get("end_time") else None,
                error=t.get("error"),
            )
            study.trials.append(trial)

        return study


class TuningTool(Tool):
    """Hyperparameter tuning with various search strategies."""

    name = "tune"
    description = """Hyperparameter tuning for ML models.

Actions:
- create: Create a new tuning study
- suggest: Get next hyperparameters to try
- report: Report trial results
- status: Get study status and best trial
- trials: List all trials
- best: Get best hyperparameters
- analyze: Analyze parameter importance

Search strategies:
- Random search (built-in)
- Grid search (built-in)
- Bayesian optimization (requires optuna)

Parameter types:
- uniform: Continuous uniform [low, high]
- log_uniform: Log-uniform [low, high] (for learning rates)
- int: Integer [low, high]
- choice: Categorical list of choices

Example:
1. create study="lr-search" space='[
     {"name": "lr", "type": "log_uniform", "low": 1e-5, "high": 1e-1},
     {"name": "batch_size", "type": "choice", "choices": [16, 32, 64]}
   ]' direction="minimize"
2. suggest study="lr-search"  (returns: {"lr": 0.001, "batch_size": 32})
3. Train model with those params...
4. report study="lr-search" trial_id=0 metric=0.85
5. Repeat steps 2-4
6. best study="lr-search\""""

    parameters = [
        ToolParameter(
            name="action",
            type=str,
            description="Action: create, suggest, report, status, trials, best, analyze",
            required=True,
        ),
        ToolParameter(
            name="study",
            type=str,
            description="Study name",
            required=True,
        ),
        ToolParameter(
            name="options",
            type=dict,
            description="Action-specific options",
            required=False,
            default={},
        ),
    ]

    def __init__(self, storage_dir: str = ".ds-agent/tuning"):
        super().__init__()
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _load_study(self, name: str) -> Study | None:
        """Load a study from disk."""
        path = self.storage_dir / f"{name}.json"
        if not path.exists():
            return None
        with open(path) as f:
            return Study.from_dict(json.load(f))

    def _save_study(self, study: Study) -> None:
        """Save a study to disk."""
        path = self.storage_dir / f"{study.name}.json"
        with open(path, "w") as f:
            json.dump(study.to_dict(), f, indent=2)

    async def execute(
        self,
        ctx: ToolContext,
        action: str,
        study: str,
        options: dict[str, Any] | None = None,
    ) -> ToolResult:
        """Execute tuning action."""
        options = options or {}
        action = action.lower()

        try:
            if action == "create":
                return self._create(study, options)
            elif action == "suggest":
                return self._suggest(study, options)
            elif action == "report":
                return self._report(study, options)
            elif action == "status":
                return self._status(study)
            elif action == "trials":
                return self._trials(study, options)
            elif action == "best":
                return self._best(study)
            elif action == "analyze":
                return self._analyze(study)
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

    def _create(self, study_name: str, options: dict[str, Any]) -> ToolResult:
        """Create a new study."""
        if self._load_study(study_name):
            if not options.get("overwrite"):
                return ToolResult(
                    tool_call_id="",
                    content=f"Study '{study_name}' already exists. Use overwrite=true to replace.",
                    is_error=True,
                )

        space_def = options.get("space", [])
        if isinstance(space_def, str):
            space_def = json.loads(space_def)

        if not space_def:
            return ToolResult(
                tool_call_id="",
                content="Search space is required. Provide 'space' as list of parameter definitions.",
                is_error=True,
            )

        space = []
        for hp in space_def:
            space.append(HyperparameterSpace(
                name=hp["name"],
                type=hp["type"],
                low=hp.get("low"),
                high=hp.get("high"),
                choices=hp.get("choices"),
            ))

        study = Study(
            name=study_name,
            space=space,
            metric_name=options.get("metric_name", "metric"),
            direction=options.get("direction", "minimize"),
        )

        self._save_study(study)

        params_desc = "\n".join(
            f"  - {hp.name}: {hp.type}" +
            (f" [{hp.low}, {hp.high}]" if hp.low is not None else "") +
            (f" {hp.choices}" if hp.choices else "")
            for hp in space
        )

        return ToolResult(
            tool_call_id="",
            content=f"""Study created: {study_name}
Direction: {study.direction}
Parameters:
{params_desc}

Use 'suggest' to get hyperparameters, 'report' to log results.""",
        )

    def _suggest(self, study_name: str, options: dict[str, Any]) -> ToolResult:
        """Suggest next hyperparameters."""
        study = self._load_study(study_name)
        if not study:
            return ToolResult(
                tool_call_id="",
                content=f"Study not found: {study_name}",
                is_error=True,
            )

        # Get next trial ID
        trial_id = len(study.trials)

        # Suggest parameters
        params = study.suggest()

        # Create trial
        trial = Trial(id=trial_id, params=params, status="running", start_time=datetime.now())
        study.trials.append(trial)
        self._save_study(study)

        params_str = json.dumps(params, indent=2)

        return ToolResult(
            tool_call_id="",
            content=f"""Trial {trial_id} suggested:
{params_str}

After training, report results with:
  report study="{study_name}" trial_id={trial_id} metric=<value>""",
            metadata={"trial_id": trial_id, "params": params},
        )

    def _report(self, study_name: str, options: dict[str, Any]) -> ToolResult:
        """Report trial results."""
        study = self._load_study(study_name)
        if not study:
            return ToolResult(
                tool_call_id="",
                content=f"Study not found: {study_name}",
                is_error=True,
            )

        trial_id = options.get("trial_id")
        if trial_id is None:
            return ToolResult(
                tool_call_id="",
                content="trial_id is required",
                is_error=True,
            )

        metric = options.get("metric")
        if metric is None:
            return ToolResult(
                tool_call_id="",
                content="metric is required",
                is_error=True,
            )

        if trial_id >= len(study.trials):
            return ToolResult(
                tool_call_id="",
                content=f"Invalid trial_id: {trial_id}",
                is_error=True,
            )

        trial = study.trials[trial_id]
        trial.metric = float(metric)
        trial.status = "completed"
        trial.end_time = datetime.now()

        self._save_study(study)

        # Check if this is the best
        is_best = study.best_trial and study.best_trial.id == trial_id
        best_str = " ⭐ NEW BEST!" if is_best else ""

        completed = sum(1 for t in study.trials if t.status == "completed")

        return ToolResult(
            tool_call_id="",
            content=f"""Trial {trial_id} completed: {study.metric_name}={metric:.6f}{best_str}
Completed trials: {completed}/{len(study.trials)}""",
        )

    def _status(self, study_name: str) -> ToolResult:
        """Get study status."""
        study = self._load_study(study_name)
        if not study:
            return ToolResult(
                tool_call_id="",
                content=f"Study not found: {study_name}",
                is_error=True,
            )

        completed = sum(1 for t in study.trials if t.status == "completed")
        running = sum(1 for t in study.trials if t.status == "running")
        failed = sum(1 for t in study.trials if t.status == "failed")

        best = study.best_trial
        best_str = ""
        if best:
            best_str = f"""
Best Trial: #{best.id}
  {study.metric_name}: {best.metric:.6f}
  Parameters: {json.dumps(best.params, indent=4)}"""

        return ToolResult(
            tool_call_id="",
            content=f"""Study: {study_name}
Direction: {study.direction}
Trials: {completed} completed, {running} running, {failed} failed
{best_str}""",
        )

    def _trials(self, study_name: str, options: dict[str, Any]) -> ToolResult:
        """List all trials."""
        study = self._load_study(study_name)
        if not study:
            return ToolResult(
                tool_call_id="",
                content=f"Study not found: {study_name}",
                is_error=True,
            )

        if not study.trials:
            return ToolResult(
                tool_call_id="",
                content="No trials yet.",
            )

        # Sort by metric
        sorted_trials = sorted(
            [t for t in study.trials if t.metric is not None],
            key=lambda t: t.metric,
            reverse=(study.direction == "maximize"),
        )

        lines = [
            f"Trials for {study_name} (sorted by {study.metric_name}):",
            "=" * 70,
            "",
        ]

        # Header
        param_names = [hp.name for hp in study.space[:4]]  # Limit to 4 params
        header = f"{'#':<4} {'Status':<10} {study.metric_name:<12}"
        for p in param_names:
            header += f" {p[:8]:<10}"
        lines.append(header)
        lines.append("-" * 70)

        for trial in sorted_trials[:20]:
            row = f"{trial.id:<4} {trial.status:<10} {trial.metric:<12.6f}" if trial.metric else f"{trial.id:<4} {trial.status:<10} {'N/A':<12}"
            for p in param_names:
                val = trial.params.get(p, "N/A")
                if isinstance(val, float):
                    row += f" {val:<10.4g}"
                else:
                    row += f" {str(val)[:9]:<10}"
            lines.append(row)

        return ToolResult(
            tool_call_id="",
            content="\n".join(lines),
        )

    def _best(self, study_name: str) -> ToolResult:
        """Get best hyperparameters."""
        study = self._load_study(study_name)
        if not study:
            return ToolResult(
                tool_call_id="",
                content=f"Study not found: {study_name}",
                is_error=True,
            )

        best = study.best_trial
        if not best:
            return ToolResult(
                tool_call_id="",
                content="No completed trials yet.",
            )

        return ToolResult(
            tool_call_id="",
            content=f"""Best Trial: #{best.id}
{study.metric_name}: {best.metric:.6f}

Parameters:
{json.dumps(best.params, indent=2)}

Use these parameters in your training code.""",
            metadata={"params": best.params, "metric": best.metric},
        )

    def _analyze(self, study_name: str) -> ToolResult:
        """Analyze parameter importance."""
        study = self._load_study(study_name)
        if not study:
            return ToolResult(
                tool_call_id="",
                content=f"Study not found: {study_name}",
                is_error=True,
            )

        completed = [t for t in study.trials if t.status == "completed" and t.metric is not None]

        if len(completed) < 5:
            return ToolResult(
                tool_call_id="",
                content="Need at least 5 completed trials for analysis.",
            )

        # Simple correlation-based importance
        lines = [
            f"Parameter Analysis for {study_name}:",
            "=" * 50,
            "",
            "Parameter correlations with metric:",
            "-" * 50,
        ]

        for hp in study.space:
            values = []
            metrics = []
            for trial in completed:
                if hp.name in trial.params:
                    val = trial.params[hp.name]
                    if isinstance(val, (int, float)):
                        values.append(val)
                        metrics.append(trial.metric)

            if len(values) >= 3:
                # Calculate correlation
                try:
                    import numpy as np
                    corr = np.corrcoef(values, metrics)[0, 1]
                    importance = abs(corr)
                    direction = "↑" if corr > 0 else "↓"
                    bar = "█" * int(importance * 20)
                    lines.append(f"  {hp.name:<20} {direction} r={corr:+.3f} [{bar}]")
                except Exception:
                    lines.append(f"  {hp.name:<20} (unable to compute)")

        # Best vs worst comparison
        if len(completed) >= 4:
            lines.extend(["", "Best vs Worst comparison:", "-" * 50])

            sorted_trials = sorted(completed, key=lambda t: t.metric)
            if study.direction == "maximize":
                sorted_trials = sorted_trials[::-1]

            best_trials = sorted_trials[:max(1, len(sorted_trials)//4)]
            worst_trials = sorted_trials[-max(1, len(sorted_trials)//4):]

            for hp in study.space:
                best_vals = [t.params.get(hp.name) for t in best_trials if hp.name in t.params]
                worst_vals = [t.params.get(hp.name) for t in worst_trials if hp.name in t.params]

                if best_vals and worst_vals:
                    if all(isinstance(v, (int, float)) for v in best_vals + worst_vals):
                        import numpy as np
                        best_mean = np.mean(best_vals)
                        worst_mean = np.mean(worst_vals)
                        lines.append(f"  {hp.name}: best avg={best_mean:.4g}, worst avg={worst_mean:.4g}")
                    else:
                        # Categorical - show mode
                        from collections import Counter
                        best_mode = Counter(best_vals).most_common(1)[0][0]
                        worst_mode = Counter(worst_vals).most_common(1)[0][0]
                        lines.append(f"  {hp.name}: best common={best_mode}, worst common={worst_mode}")

        return ToolResult(
            tool_call_id="",
            content="\n".join(lines),
        )
