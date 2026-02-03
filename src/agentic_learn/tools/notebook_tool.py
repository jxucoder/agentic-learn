"""Jupyter notebook creation and manipulation tool."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from agentic_learn.core.tool import Tool, ToolContext, ToolParameter
from agentic_learn.core.types import ToolResult


class NotebookTool(Tool):
    """Create and manipulate Jupyter notebooks."""

    name = "notebook"
    description = """Create and manipulate Jupyter notebooks programmatically.

Actions:
- create: Create a new notebook
- read: Read notebook contents
- add_cell: Add a cell (code or markdown)
- edit_cell: Edit an existing cell
- delete_cell: Delete a cell
- run: Execute notebook and capture outputs
- export: Export to Python script or HTML

Use this for:
- Creating reproducible analysis notebooks
- Building experiment reports
- Generating documentation with code
- Iterative data exploration

Notebooks support:
- Code cells with execution
- Markdown cells for documentation
- Output capture (text, images, tables)
- Kernel selection

Example:
1. create path="analysis.ipynb" title="Data Analysis"
2. add_cell path="analysis.ipynb" cell_type="code" source="import pandas as pd"
3. add_cell path="analysis.ipynb" cell_type="markdown" source="## Results"
4. run path="analysis.ipynb\""""

    parameters = [
        ToolParameter(
            name="action",
            type=str,
            description="Action: create, read, add_cell, edit_cell, delete_cell, run, export",
            required=True,
        ),
        ToolParameter(
            name="path",
            type=str,
            description="Path to the notebook file",
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

    async def execute(
        self,
        ctx: ToolContext,
        action: str,
        path: str,
        options: dict[str, Any] | None = None,
    ) -> ToolResult:
        """Execute notebook action."""
        options = options or {}
        action = action.lower()

        # Resolve path
        if not path.endswith(".ipynb"):
            path = path + ".ipynb"
        path_obj = Path(path)

        try:
            if action == "create":
                return self._create(path_obj, options)
            elif action == "read":
                return self._read(path_obj, options)
            elif action == "add_cell":
                return self._add_cell(path_obj, options)
            elif action == "edit_cell":
                return self._edit_cell(path_obj, options)
            elif action == "delete_cell":
                return self._delete_cell(path_obj, options)
            elif action == "run":
                return await self._run(path_obj, options)
            elif action == "export":
                return self._export(path_obj, options)
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

    def _create_empty_notebook(self, kernel: str = "python3") -> dict[str, Any]:
        """Create an empty notebook structure."""
        return {
            "cells": [],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": kernel,
                },
                "language_info": {
                    "name": "python",
                    "version": "3.10.0",
                },
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }

    def _create_cell(
        self,
        cell_type: str,
        source: str | list[str],
        execution_count: int | None = None,
    ) -> dict[str, Any]:
        """Create a notebook cell."""
        if isinstance(source, str):
            source = source.split("\n")

        cell: dict[str, Any] = {
            "cell_type": cell_type,
            "metadata": {},
            "source": source,
        }

        if cell_type == "code":
            cell["execution_count"] = execution_count
            cell["outputs"] = []

        return cell

    def _load_notebook(self, path: Path) -> dict[str, Any]:
        """Load a notebook from disk."""
        with open(path) as f:
            return json.load(f)

    def _save_notebook(self, path: Path, notebook: dict[str, Any]) -> None:
        """Save a notebook to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(notebook, f, indent=2)

    def _create(self, path: Path, options: dict[str, Any]) -> ToolResult:
        """Create a new notebook."""
        if path.exists() and not options.get("overwrite"):
            return ToolResult(
                tool_call_id="",
                content=f"Notebook already exists: {path}. Use overwrite=true to replace.",
                is_error=True,
            )

        kernel = options.get("kernel", "python3")
        notebook = self._create_empty_notebook(kernel)

        # Add title cell if provided
        title = options.get("title")
        if title:
            notebook["cells"].append(
                self._create_cell("markdown", f"# {title}")
            )

        # Add description cell if provided
        description = options.get("description")
        if description:
            notebook["cells"].append(
                self._create_cell("markdown", description)
            )

        # Add initial imports cell
        if options.get("add_imports", True):
            imports = options.get("imports", [
                "import numpy as np",
                "import pandas as pd",
                "import matplotlib.pyplot as plt",
                "%matplotlib inline",
            ])
            notebook["cells"].append(
                self._create_cell("code", "\n".join(imports))
            )

        self._save_notebook(path, notebook)

        return ToolResult(
            tool_call_id="",
            content=f"""Notebook created: {path}
  Cells: {len(notebook['cells'])}
  Kernel: {kernel}""",
        )

    def _read(self, path: Path, options: dict[str, Any]) -> ToolResult:
        """Read notebook contents."""
        if not path.exists():
            return ToolResult(
                tool_call_id="",
                content=f"Notebook not found: {path}",
                is_error=True,
            )

        notebook = self._load_notebook(path)
        cells = notebook.get("cells", [])

        lines = [f"Notebook: {path}", f"Cells: {len(cells)}", "=" * 60, ""]

        for i, cell in enumerate(cells):
            cell_type = cell.get("cell_type", "unknown")
            source = cell.get("source", [])
            if isinstance(source, list):
                source = "".join(source)

            lines.append(f"[{i}] {cell_type.upper()}")
            lines.append("-" * 40)

            # Truncate long cells
            if len(source) > 500:
                source = source[:500] + "\n... (truncated)"

            lines.append(source)

            # Show outputs for code cells
            if cell_type == "code" and cell.get("outputs"):
                lines.append("\nOutput:")
                for output in cell["outputs"][:3]:  # Limit outputs
                    if output.get("text"):
                        text = "".join(output["text"])
                        if len(text) > 200:
                            text = text[:200] + "..."
                        lines.append(f"  {text}")
                    elif output.get("data"):
                        lines.append("  [rich output]")

            lines.append("")

        return ToolResult(
            tool_call_id="",
            content="\n".join(lines),
        )

    def _add_cell(self, path: Path, options: dict[str, Any]) -> ToolResult:
        """Add a cell to the notebook."""
        if not path.exists():
            return ToolResult(
                tool_call_id="",
                content=f"Notebook not found: {path}",
                is_error=True,
            )

        cell_type = options.get("cell_type", "code")
        source = options.get("source", "")
        position = options.get("position")  # None = append

        notebook = self._load_notebook(path)
        cell = self._create_cell(cell_type, source)

        if position is not None:
            notebook["cells"].insert(position, cell)
        else:
            notebook["cells"].append(cell)

        self._save_notebook(path, notebook)

        cell_index = position if position is not None else len(notebook["cells"]) - 1

        return ToolResult(
            tool_call_id="",
            content=f"Added {cell_type} cell at position {cell_index}",
        )

    def _edit_cell(self, path: Path, options: dict[str, Any]) -> ToolResult:
        """Edit an existing cell."""
        if not path.exists():
            return ToolResult(
                tool_call_id="",
                content=f"Notebook not found: {path}",
                is_error=True,
            )

        cell_index = options.get("cell_index")
        if cell_index is None:
            return ToolResult(
                tool_call_id="",
                content="cell_index is required",
                is_error=True,
            )

        source = options.get("source")
        if source is None:
            return ToolResult(
                tool_call_id="",
                content="source is required",
                is_error=True,
            )

        notebook = self._load_notebook(path)

        if cell_index < 0 or cell_index >= len(notebook["cells"]):
            return ToolResult(
                tool_call_id="",
                content=f"Invalid cell index: {cell_index}",
                is_error=True,
            )

        if isinstance(source, str):
            source = source.split("\n")

        notebook["cells"][cell_index]["source"] = source
        # Clear outputs if it's a code cell
        if notebook["cells"][cell_index].get("cell_type") == "code":
            notebook["cells"][cell_index]["outputs"] = []
            notebook["cells"][cell_index]["execution_count"] = None

        self._save_notebook(path, notebook)

        return ToolResult(
            tool_call_id="",
            content=f"Updated cell {cell_index}",
        )

    def _delete_cell(self, path: Path, options: dict[str, Any]) -> ToolResult:
        """Delete a cell from the notebook."""
        if not path.exists():
            return ToolResult(
                tool_call_id="",
                content=f"Notebook not found: {path}",
                is_error=True,
            )

        cell_index = options.get("cell_index")
        if cell_index is None:
            return ToolResult(
                tool_call_id="",
                content="cell_index is required",
                is_error=True,
            )

        notebook = self._load_notebook(path)

        if cell_index < 0 or cell_index >= len(notebook["cells"]):
            return ToolResult(
                tool_call_id="",
                content=f"Invalid cell index: {cell_index}",
                is_error=True,
            )

        del notebook["cells"][cell_index]
        self._save_notebook(path, notebook)

        return ToolResult(
            tool_call_id="",
            content=f"Deleted cell {cell_index}",
        )

    async def _run(self, path: Path, options: dict[str, Any]) -> ToolResult:
        """Execute the notebook."""
        if not path.exists():
            return ToolResult(
                tool_call_id="",
                content=f"Notebook not found: {path}",
                is_error=True,
            )

        try:
            import nbformat
            from nbconvert.preprocessors import ExecutePreprocessor
        except ImportError:
            return ToolResult(
                tool_call_id="",
                content="nbconvert required: pip install nbconvert",
                is_error=True,
            )

        # Load notebook
        with open(path) as f:
            nb = nbformat.read(f, as_version=4)

        # Execute
        ep = ExecutePreprocessor(
            timeout=options.get("timeout", 600),
            kernel_name=options.get("kernel", "python3"),
        )

        try:
            ep.preprocess(nb, {"metadata": {"path": str(path.parent)}})

            # Save executed notebook
            output_path = options.get("output", path)
            with open(output_path, "w") as f:
                nbformat.write(nb, f)

            # Count successful cells
            executed = sum(
                1 for cell in nb.cells
                if cell.cell_type == "code" and cell.execution_count
            )

            return ToolResult(
                tool_call_id="",
                content=f"""Notebook executed successfully: {path}
  Cells executed: {executed}
  Output saved to: {output_path}""",
            )

        except Exception as e:
            return ToolResult(
                tool_call_id="",
                content=f"Notebook execution failed: {str(e)}",
                is_error=True,
            )

    def _export(self, path: Path, options: dict[str, Any]) -> ToolResult:
        """Export notebook to another format."""
        if not path.exists():
            return ToolResult(
                tool_call_id="",
                content=f"Notebook not found: {path}",
                is_error=True,
            )

        format = options.get("format", "python")
        output = options.get("output")

        notebook = self._load_notebook(path)

        if format == "python":
            # Export to Python script
            lines = [f"# Exported from {path}", ""]

            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    source = cell.get("source", [])
                    if isinstance(source, list):
                        source = "".join(source)
                    lines.append(source)
                    lines.append("")
                elif cell["cell_type"] == "markdown":
                    source = cell.get("source", [])
                    if isinstance(source, list):
                        source = "".join(source)
                    # Convert to comments
                    for line in source.split("\n"):
                        lines.append(f"# {line}")
                    lines.append("")

            content = "\n".join(lines)

            if output:
                output_path = Path(output)
                with open(output_path, "w") as f:
                    f.write(content)
                return ToolResult(
                    tool_call_id="",
                    content=f"Exported to {output_path}",
                )
            else:
                return ToolResult(
                    tool_call_id="",
                    content=f"Python export:\n{content[:2000]}",
                )

        elif format == "html":
            try:
                import nbformat
                from nbconvert import HTMLExporter
            except ImportError:
                return ToolResult(
                    tool_call_id="",
                    content="nbconvert required: pip install nbconvert",
                    is_error=True,
                )

            with open(path) as f:
                nb = nbformat.read(f, as_version=4)

            exporter = HTMLExporter()
            html, _ = exporter.from_notebook_node(nb)

            output_path = Path(output) if output else path.with_suffix(".html")
            with open(output_path, "w") as f:
                f.write(html)

            return ToolResult(
                tool_call_id="",
                content=f"Exported to {output_path}",
            )

        else:
            return ToolResult(
                tool_call_id="",
                content=f"Unknown format: {format}. Supported: python, html",
                is_error=True,
            )
