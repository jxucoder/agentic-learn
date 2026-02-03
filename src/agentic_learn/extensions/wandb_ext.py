"""Weights & Biases extension for experiment tracking."""

from __future__ import annotations

import json
from typing import Any

from agentic_learn.core.extension import Extension, ExtensionAPI
from agentic_learn.core.tool import Tool, ToolContext, ToolParameter
from agentic_learn.core.types import EventType, ToolResult


class WandbTool(Tool):
    """Weights & Biases experiment tracking integration."""

    name = "wandb"
    description = """Weights & Biases (W&B) experiment tracking.

Actions:
- init: Initialize a W&B run
- log: Log metrics to W&B
- config: Set/update run config
- summary: Set run summary metrics
- artifact: Log an artifact
- finish: Finish the current run
- status: Get current run status
- runs: List recent runs in a project

Features:
- Automatic metric visualization
- Hyperparameter tracking
- Model checkpointing
- Team collaboration

Requires: wandb package and WANDB_API_KEY env var

Examples:
- init project="my-project" name="experiment-1"
- log loss=0.5 accuracy=0.85 step=100
- artifact path="model.pt" type="model\""""

    parameters = [
        ToolParameter(
            name="action",
            type=str,
            description="Action: init, log, config, summary, artifact, finish, status, runs",
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

    def __init__(self):
        super().__init__()
        self._run: Any = None

    async def execute(
        self,
        ctx: ToolContext,
        action: str,
        options: dict[str, Any] | None = None,
    ) -> ToolResult:
        """Execute W&B action."""
        options = options or {}

        try:
            import wandb
        except ImportError:
            return ToolResult(
                tool_call_id="",
                content="W&B not installed. Install with: pip install wandb",
                is_error=True,
            )

        action = action.lower()

        try:
            if action == "init":
                return self._init_run(wandb, options)
            elif action == "log":
                return self._log_metrics(wandb, options)
            elif action == "config":
                return self._update_config(wandb, options)
            elif action == "summary":
                return self._set_summary(wandb, options)
            elif action == "artifact":
                return await self._log_artifact(wandb, options)
            elif action == "finish":
                return self._finish_run(wandb)
            elif action == "status":
                return self._get_status(wandb)
            elif action == "runs":
                return self._list_runs(wandb, options)
            else:
                return ToolResult(
                    tool_call_id="",
                    content=f"Unknown action: {action}",
                    is_error=True,
                )
        except Exception as e:
            return ToolResult(
                tool_call_id="",
                content=f"W&B error: {str(e)}",
                is_error=True,
            )

    def _init_run(self, wandb: Any, options: dict[str, Any]) -> ToolResult:
        """Initialize a W&B run."""
        if wandb.run is not None:
            return ToolResult(
                tool_call_id="",
                content=f"Run already active: {wandb.run.name} ({wandb.run.id})",
            )

        project = options.get("project", "ds-agent")
        name = options.get("name")
        config = options.get("config", {})
        tags = options.get("tags", [])
        notes = options.get("notes", "")

        self._run = wandb.init(
            project=project,
            name=name,
            config=config,
            tags=tags,
            notes=notes,
            reinit=True,
        )

        return ToolResult(
            tool_call_id="",
            content=f"""W&B run initialized:
  Project: {project}
  Run Name: {self._run.name}
  Run ID: {self._run.id}
  URL: {self._run.url}""",
            metadata={"run_id": self._run.id, "url": self._run.url},
        )

    def _log_metrics(self, wandb: Any, options: dict[str, Any]) -> ToolResult:
        """Log metrics to W&B."""
        if wandb.run is None:
            return ToolResult(
                tool_call_id="",
                content="No active run. Use action='init' first.",
                is_error=True,
            )

        step = options.pop("step", None)
        commit = options.pop("commit", True)

        # Filter to only numeric values
        metrics = {k: v for k, v in options.items() if isinstance(v, (int, float))}

        if not metrics:
            return ToolResult(
                tool_call_id="",
                content="No metrics to log (provide numeric key=value pairs).",
                is_error=True,
            )

        wandb.log(metrics, step=step, commit=commit)

        return ToolResult(
            tool_call_id="",
            content=f"Logged metrics: {json.dumps(metrics)}" + (f" (step={step})" if step else ""),
        )

    def _update_config(self, wandb: Any, options: dict[str, Any]) -> ToolResult:
        """Update run config."""
        if wandb.run is None:
            return ToolResult(
                tool_call_id="",
                content="No active run. Use action='init' first.",
                is_error=True,
            )

        wandb.config.update(options)

        return ToolResult(
            tool_call_id="",
            content=f"Updated config: {json.dumps(options)}",
        )

    def _set_summary(self, wandb: Any, options: dict[str, Any]) -> ToolResult:
        """Set run summary metrics."""
        if wandb.run is None:
            return ToolResult(
                tool_call_id="",
                content="No active run. Use action='init' first.",
                is_error=True,
            )

        for key, value in options.items():
            wandb.run.summary[key] = value

        return ToolResult(
            tool_call_id="",
            content=f"Set summary: {json.dumps(options)}",
        )

    async def _log_artifact(self, wandb: Any, options: dict[str, Any]) -> ToolResult:
        """Log an artifact."""
        if wandb.run is None:
            return ToolResult(
                tool_call_id="",
                content="No active run. Use action='init' first.",
                is_error=True,
            )

        path = options.get("path")
        artifact_type = options.get("type", "file")
        name = options.get("name")

        if not path:
            return ToolResult(
                tool_call_id="",
                content="Artifact path required.",
                is_error=True,
            )

        artifact = wandb.Artifact(
            name=name or path.replace("/", "_"),
            type=artifact_type,
        )
        artifact.add_file(path)
        wandb.log_artifact(artifact)

        return ToolResult(
            tool_call_id="",
            content=f"Logged artifact: {path} (type={artifact_type})",
        )

    def _finish_run(self, wandb: Any) -> ToolResult:
        """Finish the current run."""
        if wandb.run is None:
            return ToolResult(
                tool_call_id="",
                content="No active run to finish.",
            )

        run_name = wandb.run.name
        run_url = wandb.run.url

        wandb.finish()
        self._run = None

        return ToolResult(
            tool_call_id="",
            content=f"""Run finished: {run_name}
View at: {run_url}""",
        )

    def _get_status(self, wandb: Any) -> ToolResult:
        """Get current run status."""
        if wandb.run is None:
            return ToolResult(
                tool_call_id="",
                content="No active W&B run.",
            )

        run = wandb.run
        summary = dict(run.summary) if hasattr(run.summary, "items") else {}

        return ToolResult(
            tool_call_id="",
            content=f"""W&B Run Status:
  Name: {run.name}
  ID: {run.id}
  Project: {run.project}
  URL: {run.url}
  Config: {json.dumps(dict(run.config), indent=2)}
  Summary: {json.dumps(summary, indent=2)}""",
        )

    def _list_runs(self, wandb: Any, options: dict[str, Any]) -> ToolResult:
        """List recent runs in a project."""
        project = options.get("project")
        entity = options.get("entity")
        limit = options.get("limit", 10)

        if not project:
            if wandb.run:
                project = wandb.run.project
                entity = wandb.run.entity
            else:
                return ToolResult(
                    tool_call_id="",
                    content="Project name required.",
                    is_error=True,
                )

        api = wandb.Api()

        try:
            path = f"{entity}/{project}" if entity else project
            runs = api.runs(path, per_page=limit)

            lines = [f"Recent runs in '{project}':", "=" * 60, ""]
            lines.append(f"{'Name':<25} {'State':<12} {'Created'}")
            lines.append("-" * 60)

            for run in runs:
                created = run.created_at[:16] if run.created_at else "Unknown"
                lines.append(f"{run.name[:24]:<25} {run.state:<12} {created}")

            return ToolResult(
                tool_call_id="",
                content="\n".join(lines),
            )

        except Exception as e:
            return ToolResult(
                tool_call_id="",
                content=f"Failed to list runs: {e}",
                is_error=True,
            )


class WandbExtension(Extension):
    """Extension for Weights & Biases experiment tracking."""

    name = "wandb"
    description = "Weights & Biases experiment tracking integration"
    version = "0.1.0"

    def setup(self, api: ExtensionAPI) -> None:
        """Register W&B tools and hooks."""
        api.register_tool(WandbTool())

        # Auto-log agent events to W&B if run is active
        api.on(EventType.TOOL_CALL_END, self._on_tool_call)

        # Register commands
        api.register_command(
            "wandb",
            "W&B management: /wandb <init|status|finish>",
            self._wandb_command,
        )

    async def _on_tool_call(self, ctx: Any) -> None:
        """Log tool calls to W&B."""
        try:
            import wandb

            if wandb.run is not None and ctx.event:
                data = ctx.event.data
                if data.get("tool") and not data.get("is_error"):
                    # Log tool usage
                    wandb.log({
                        "tool_calls": 1,
                        f"tool_{data['tool']}": 1,
                    })
        except Exception:
            pass  # Don't fail on logging errors

    async def _wandb_command(self, ctx: Any, args: list[str]) -> None:
        """Handle /wandb command."""
        action = args[0] if args else "status"
        tool = WandbTool()
        result = await tool.execute(ctx, action=action)
        print(result.content)
