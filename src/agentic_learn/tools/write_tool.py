"""File writing tool."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from agentic_learn.core.tool import Tool, ToolContext, ToolParameter
from agentic_learn.core.types import ToolResult


class WriteTool(Tool):
    """Write content to files."""

    name = "write"
    description = """Write content to a file, creating it if it doesn't exist.

Features:
- Creates parent directories automatically
- Overwrites existing files (use edit for modifications)
- Supports any text file type

Use this to:
- Create new Python scripts
- Write configuration files
- Create data files
- Generate reports

Safety:
- Will warn before overwriting existing files
- Use overwrite=true to confirm overwrites

Examples:
- write path="model.py" content="import torch\\n..."
- write path="config.yaml" content="lr: 0.001"
- write path="src/utils/helpers.py" content="..." (creates directories)"""

    parameters = [
        ToolParameter(
            name="path",
            type=str,
            description="Path to write the file to",
            required=True,
        ),
        ToolParameter(
            name="content",
            type=str,
            description="Content to write to the file",
            required=True,
        ),
        ToolParameter(
            name="overwrite",
            type=bool,
            description="Whether to overwrite existing files (default: true)",
            required=False,
            default=True,
        ),
    ]

    async def execute(
        self,
        ctx: ToolContext,
        path: str,
        content: str,
        overwrite: bool = True,
    ) -> ToolResult:
        """Write content to a file."""
        # Resolve path
        path_obj = Path(path)
        if not path_obj.is_absolute():
            path_obj = Path(ctx.cwd) / path_obj

        # Check if file exists
        exists = path_obj.exists()
        if exists and not overwrite:
            return ToolResult(
                tool_call_id="",
                content=f"File already exists: {path}. Use overwrite=true to replace.",
                is_error=True,
            )

        # Check if path is a directory
        if path_obj.is_dir():
            return ToolResult(
                tool_call_id="",
                content=f"Path is a directory: {path}",
                is_error=True,
            )

        try:
            # Create parent directories
            path_obj.parent.mkdir(parents=True, exist_ok=True)

            # Write content
            with open(path_obj, "w", encoding="utf-8") as f:
                f.write(content)

            # Get file stats
            size = path_obj.stat().st_size
            lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)

            action = "Overwrote" if exists else "Created"

            return ToolResult(
                tool_call_id="",
                content=f"{action}: {path}\n  Lines: {lines}\n  Size: {size} bytes",
                metadata={"path": str(path_obj), "lines": lines, "size": size},
            )

        except PermissionError:
            return ToolResult(
                tool_call_id="",
                content=f"Permission denied: {path}",
                is_error=True,
            )
        except OSError as e:
            return ToolResult(
                tool_call_id="",
                content=f"Error writing file: {e}",
                is_error=True,
            )
