"""Enhanced visualization tool for data science."""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import Any

from agentic_learn.core.tool import Tool, ToolContext, ToolParameter
from agentic_learn.core.types import ToolResult


class VizTool(Tool):
    """Create visualizations for data analysis and ML experiments."""

    name = "viz"
    description = """Create visualizations for data analysis and ML.

Plot types:
- line: Line plots for time series, training curves
- scatter: Scatter plots for relationships
- histogram: Distribution analysis
- bar: Categorical comparisons
- heatmap: Correlation matrices, confusion matrices
- box: Distribution comparison across groups
- pair: Pairwise relationships (seaborn pairplot)
- learning_curve: Training/validation metrics over time
- feature_importance: Model feature importances
- confusion: Confusion matrix visualization
- distribution: Multiple distributions comparison
- residuals: Model residual analysis

Actions:
- plot: Create a plot
- save: Save current figure
- show: Display plot info (path)
- clear: Clear current figure
- style: Set plot style

Output:
- Saves plots to .ds-agent/plots/ by default
- Returns path to saved image
- Supports PNG, PDF, SVG formats

Example:
  viz action="plot" type="line" data='{"x": [1,2,3], "y": [1,4,9]}' title="Training Loss"
  viz action="plot" type="learning_curve" data='{"train": [...], "val": [...]}'
  viz action="plot" type="confusion" data='{"matrix": [[50,10],[5,35]], "labels": ["cat","dog"]}'"""

    parameters = [
        ToolParameter(
            name="action",
            type=str,
            description="Action: plot, save, show, clear, style",
            required=True,
        ),
        ToolParameter(
            name="options",
            type=dict,
            description="Plot options (type, data, title, xlabel, ylabel, etc.)",
            required=False,
            default={},
        ),
    ]

    def __init__(self, output_dir: str = ".ds-agent/plots"):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._figure_counter = 0

    async def execute(
        self,
        ctx: ToolContext,
        action: str,
        options: dict[str, Any] | None = None,
    ) -> ToolResult:
        """Execute visualization action."""
        options = options or {}

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return ToolResult(
                tool_call_id="",
                content="matplotlib required: pip install matplotlib",
                is_error=True,
            )

        action = action.lower()

        try:
            if action == "plot":
                return self._plot(plt, options)
            elif action == "save":
                return self._save(plt, options)
            elif action == "show":
                return self._show(plt)
            elif action == "clear":
                plt.clf()
                plt.close("all")
                return ToolResult(tool_call_id="", content="Figures cleared.")
            elif action == "style":
                return self._style(plt, options)
            else:
                return ToolResult(
                    tool_call_id="",
                    content=f"Unknown action: {action}",
                    is_error=True,
                )
        except Exception as e:
            import traceback
            return ToolResult(
                tool_call_id="",
                content=f"Error: {str(e)}\n{traceback.format_exc()}",
                is_error=True,
            )

    def _plot(self, plt: Any, options: dict[str, Any]) -> ToolResult:
        """Create a plot."""
        plot_type = options.get("type", "line")
        data = options.get("data", {})
        if isinstance(data, str):
            data = json.loads(data)

        title = options.get("title", "")
        xlabel = options.get("xlabel", "")
        ylabel = options.get("ylabel", "")
        figsize = options.get("figsize", (10, 6))
        save = options.get("save", True)
        filename = options.get("filename")

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Dispatch to plot type
        plot_func = getattr(self, f"_plot_{plot_type}", None)
        if plot_func is None:
            return ToolResult(
                tool_call_id="",
                content=f"Unknown plot type: {plot_type}",
                is_error=True,
            )

        plot_func(ax, data, options)

        # Set labels
        if title:
            ax.set_title(title, fontsize=14)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

        # Legend if needed
        if options.get("legend", True) and ax.get_legend_handles_labels()[0]:
            ax.legend()

        plt.tight_layout()

        # Save
        if save:
            self._figure_counter += 1
            if filename:
                path = self.output_dir / filename
            else:
                path = self.output_dir / f"plot_{self._figure_counter:04d}.png"

            path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(path, dpi=options.get("dpi", 150), bbox_inches="tight")
            plt.close(fig)

            return ToolResult(
                tool_call_id="",
                content=f"Plot saved: {path}",
                metadata={"path": str(path)},
            )

        return ToolResult(
            tool_call_id="",
            content="Plot created (not saved).",
        )

    def _plot_line(self, ax: Any, data: dict, options: dict) -> None:
        """Line plot."""
        x = data.get("x")
        y = data.get("y")

        if isinstance(y, dict):
            # Multiple lines
            for label, values in y.items():
                if x:
                    ax.plot(x, values, label=label, marker=options.get("marker"))
                else:
                    ax.plot(values, label=label, marker=options.get("marker"))
        elif y is not None:
            if x:
                ax.plot(x, y, marker=options.get("marker"))
            else:
                ax.plot(y, marker=options.get("marker"))

        if options.get("grid", True):
            ax.grid(True, alpha=0.3)

    def _plot_scatter(self, ax: Any, data: dict, options: dict) -> None:
        """Scatter plot."""
        x = data.get("x", [])
        y = data.get("y", [])
        c = data.get("c") or data.get("color")
        s = data.get("s") or data.get("size", 50)

        scatter = ax.scatter(x, y, c=c, s=s, alpha=options.get("alpha", 0.7))

        if c is not None and options.get("colorbar", True):
            plt = ax.figure.canvas.manager
            ax.figure.colorbar(scatter, ax=ax)

    def _plot_histogram(self, ax: Any, data: dict, options: dict) -> None:
        """Histogram."""
        values = data.get("values") or data.get("x", [])
        bins = options.get("bins", "auto")

        if isinstance(values, dict):
            for label, vals in values.items():
                ax.hist(vals, bins=bins, alpha=0.7, label=label)
        else:
            ax.hist(values, bins=bins, alpha=0.7, edgecolor="black")

    def _plot_bar(self, ax: Any, data: dict, options: dict) -> None:
        """Bar plot."""
        x = data.get("x", data.get("labels", []))
        y = data.get("y", data.get("values", []))

        if isinstance(y, dict):
            # Grouped bar
            import numpy as np
            n_groups = len(x)
            n_bars = len(y)
            width = 0.8 / n_bars
            indices = np.arange(n_groups)

            for i, (label, values) in enumerate(y.items()):
                ax.bar(indices + i * width, values, width, label=label)

            ax.set_xticks(indices + width * (n_bars - 1) / 2)
            ax.set_xticklabels(x)
        else:
            ax.bar(x, y)
            if options.get("rotate_labels"):
                ax.tick_params(axis="x", rotation=45)

    def _plot_heatmap(self, ax: Any, data: dict, options: dict) -> None:
        """Heatmap."""
        import numpy as np

        matrix = np.array(data.get("matrix", data.get("values", [])))
        labels_x = data.get("labels_x") or data.get("columns")
        labels_y = data.get("labels_y") or data.get("index") or data.get("rows")

        cmap = options.get("cmap", "viridis")
        im = ax.imshow(matrix, cmap=cmap, aspect="auto")

        # Add colorbar
        ax.figure.colorbar(im, ax=ax)

        # Add labels
        if labels_x:
            ax.set_xticks(range(len(labels_x)))
            ax.set_xticklabels(labels_x, rotation=45, ha="right")
        if labels_y:
            ax.set_yticks(range(len(labels_y)))
            ax.set_yticklabels(labels_y)

        # Add values
        if options.get("annotate", True) and matrix.size <= 100:
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    val = matrix[i, j]
                    text = f"{val:.2f}" if isinstance(val, float) else str(val)
                    ax.text(j, i, text, ha="center", va="center",
                           color="white" if val > matrix.mean() else "black")

    def _plot_box(self, ax: Any, data: dict, options: dict) -> None:
        """Box plot."""
        values = data.get("values") or data.get("data", {})
        labels = data.get("labels")

        if isinstance(values, dict):
            ax.boxplot(list(values.values()), labels=list(values.keys()))
        elif isinstance(values, list):
            ax.boxplot(values, labels=labels)

    def _plot_learning_curve(self, ax: Any, data: dict, options: dict) -> None:
        """Learning curve plot (training + validation)."""
        train = data.get("train", [])
        val = data.get("val") or data.get("validation", [])
        epochs = data.get("epochs") or list(range(1, len(train) + 1))

        ax.plot(epochs, train, label="Training", marker="o", markersize=3)
        if val:
            ax.plot(epochs, val, label="Validation", marker="s", markersize=3)

        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)

        # Find best validation point
        if val:
            best_epoch = epochs[val.index(min(val))]
            best_val = min(val)
            ax.axvline(x=best_epoch, color="gray", linestyle="--", alpha=0.5)
            ax.annotate(f"Best: {best_val:.4f}", xy=(best_epoch, best_val),
                       xytext=(5, 5), textcoords="offset points")

    def _plot_feature_importance(self, ax: Any, data: dict, options: dict) -> None:
        """Feature importance plot."""
        features = data.get("features", data.get("names", []))
        importance = data.get("importance", data.get("values", []))

        # Sort by importance
        import numpy as np
        indices = np.argsort(importance)
        if options.get("top_n"):
            indices = indices[-options["top_n"]:]

        sorted_features = [features[i] for i in indices]
        sorted_importance = [importance[i] for i in indices]

        ax.barh(sorted_features, sorted_importance)
        ax.set_xlabel("Importance")

    def _plot_confusion(self, ax: Any, data: dict, options: dict) -> None:
        """Confusion matrix."""
        import numpy as np

        matrix = np.array(data.get("matrix", []))
        labels = data.get("labels", [])

        # Normalize if requested
        if options.get("normalize"):
            matrix = matrix.astype(float) / matrix.sum(axis=1, keepdims=True)

        cmap = options.get("cmap", "Blues")
        im = ax.imshow(matrix, cmap=cmap)
        ax.figure.colorbar(im, ax=ax)

        if labels:
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels)

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        # Annotate
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                text = f"{val:.2f}" if options.get("normalize") else str(int(val))
                color = "white" if val > matrix.max() / 2 else "black"
                ax.text(j, i, text, ha="center", va="center", color=color)

    def _plot_distribution(self, ax: Any, data: dict, options: dict) -> None:
        """Distribution comparison with KDE."""
        try:
            import seaborn as sns
            use_seaborn = True
        except ImportError:
            use_seaborn = False

        values = data.get("values", {})
        if not isinstance(values, dict):
            values = {"data": values}

        for label, vals in values.items():
            if use_seaborn:
                sns.kdeplot(vals, ax=ax, label=label, fill=True, alpha=0.3)
            else:
                ax.hist(vals, bins=30, alpha=0.5, density=True, label=label)

    def _plot_residuals(self, ax: Any, data: dict, options: dict) -> None:
        """Residual plot for regression."""
        predicted = data.get("predicted", data.get("y_pred", []))
        actual = data.get("actual", data.get("y_true", []))

        import numpy as np
        residuals = np.array(actual) - np.array(predicted)

        ax.scatter(predicted, residuals, alpha=0.5)
        ax.axhline(y=0, color="red", linestyle="--")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residuals")

        # Add residual statistics
        rmse = np.sqrt(np.mean(residuals**2))
        ax.text(0.02, 0.98, f"RMSE: {rmse:.4f}", transform=ax.transAxes,
               verticalalignment="top")

    def _save(self, plt: Any, options: dict) -> ToolResult:
        """Save current figure."""
        filename = options.get("filename", f"figure_{self._figure_counter + 1:04d}.png")
        path = self.output_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)

        dpi = options.get("dpi", 150)
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
        self._figure_counter += 1

        return ToolResult(
            tool_call_id="",
            content=f"Figure saved: {path}",
            metadata={"path": str(path)},
        )

    def _show(self, plt: Any) -> ToolResult:
        """Show info about current figure."""
        fig = plt.gcf()
        axes = fig.axes

        return ToolResult(
            tool_call_id="",
            content=f"""Current figure:
  Size: {fig.get_size_inches()}
  Axes: {len(axes)}
  Output dir: {self.output_dir}""",
        )

    def _style(self, plt: Any, options: dict) -> ToolResult:
        """Set plot style."""
        style = options.get("style", "seaborn-v0_8-whitegrid")

        available = plt.style.available
        if style not in available:
            return ToolResult(
                tool_call_id="",
                content=f"Style '{style}' not found. Available: {available}",
                is_error=True,
            )

        plt.style.use(style)

        return ToolResult(
            tool_call_id="",
            content=f"Style set to: {style}",
        )
