"""File reading tool."""

from __future__ import annotations

import mimetypes
import os
from pathlib import Path
from typing import Any

from agentic_learn.core.tool import Tool, ToolContext, ToolParameter
from agentic_learn.core.types import ToolResult


class ReadTool(Tool):
    """Read files from the filesystem."""

    name = "read"
    description = """Read file contents from the filesystem.

Supports:
- Text files (Python, JSON, YAML, Markdown, etc.)
- Binary files (returns info, not content)
- Partial reads with offset and limit
- Directory listing (when path is a directory)

Use this to:
- Read source code files
- Examine configuration files
- Check data file contents
- List directory contents

Examples:
- read path="model.py"
- read path="config.yaml"
- read path="src/" (lists directory)
- read path="large_file.py" offset=100 limit=50"""

    parameters = [
        ToolParameter(
            name="path",
            type=str,
            description="Path to the file or directory to read",
            required=True,
        ),
        ToolParameter(
            name="offset",
            type=int,
            description="Line number to start reading from (1-indexed, default: 1)",
            required=False,
            default=1,
        ),
        ToolParameter(
            name="limit",
            type=int,
            description="Maximum number of lines to read (default: 2000)",
            required=False,
            default=2000,
        ),
    ]

    async def execute(
        self,
        ctx: ToolContext,
        path: str,
        offset: int = 1,
        limit: int = 2000,
    ) -> ToolResult:
        """Read a file or directory."""
        # Resolve path
        path_obj = Path(path)
        if not path_obj.is_absolute():
            path_obj = Path(ctx.cwd) / path_obj

        if not path_obj.exists():
            return ToolResult(
                tool_call_id="",
                content=f"Path not found: {path}",
                is_error=True,
            )

        # Handle directory
        if path_obj.is_dir():
            return self._list_directory(path_obj)

        # Handle file
        return self._read_file(path_obj, offset, limit)

    def _list_directory(self, path: Path) -> ToolResult:
        """List directory contents."""
        try:
            entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))

            lines = [f"Directory: {path}", ""]

            for entry in entries[:100]:  # Limit to 100 entries
                if entry.is_dir():
                    lines.append(f"  📁 {entry.name}/")
                else:
                    size = entry.stat().st_size
                    if size < 1024:
                        size_str = f"{size}B"
                    elif size < 1024 * 1024:
                        size_str = f"{size/1024:.1f}KB"
                    else:
                        size_str = f"{size/1024/1024:.1f}MB"
                    lines.append(f"  📄 {entry.name} ({size_str})")

            if len(list(path.iterdir())) > 100:
                lines.append(f"  ... and {len(list(path.iterdir())) - 100} more")

            return ToolResult(
                tool_call_id="",
                content="\n".join(lines),
            )
        except PermissionError:
            return ToolResult(
                tool_call_id="",
                content=f"Permission denied: {path}",
                is_error=True,
            )

    def _read_file(self, path: Path, offset: int, limit: int) -> ToolResult:
        """Read file contents."""
        # Check if binary
        mime_type, _ = mimetypes.guess_type(str(path))
        is_binary = mime_type and not mime_type.startswith("text") and mime_type not in [
            "application/json",
            "application/xml",
            "application/javascript",
            "application/x-python-code",
        ]

        # Check file size
        file_size = path.stat().st_size
        if file_size > 10 * 1024 * 1024:  # 10MB
            return ToolResult(
                tool_call_id="",
                content=f"File too large: {file_size / 1024 / 1024:.1f}MB. Use offset/limit for partial reads.",
                is_error=True,
            )

        if is_binary:
            return ToolResult(
                tool_call_id="",
                content=f"Binary file: {path}\nSize: {file_size} bytes\nType: {mime_type}",
            )

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            total_lines = len(lines)

            # Apply offset and limit
            start = max(0, offset - 1)
            end = min(total_lines, start + limit)
            selected_lines = lines[start:end]

            # Format with line numbers
            formatted = []
            for i, line in enumerate(selected_lines, start=start + 1):
                # Truncate very long lines
                if len(line) > 500:
                    line = line[:500] + "... (truncated)\n"
                formatted.append(f"{i:6d}\t{line.rstrip()}")

            content = "\n".join(formatted)

            # Add header info
            header = f"File: {path} ({total_lines} lines)"
            if start > 0 or end < total_lines:
                header += f"\nShowing lines {start + 1}-{end}"

            return ToolResult(
                tool_call_id="",
                content=f"{header}\n{'─' * 60}\n{content}",
                metadata={"total_lines": total_lines, "shown": len(selected_lines)},
            )

        except UnicodeDecodeError:
            return ToolResult(
                tool_call_id="",
                content=f"Cannot decode file as text: {path}",
                is_error=True,
            )
        except PermissionError:
            return ToolResult(
                tool_call_id="",
                content=f"Permission denied: {path}",
                is_error=True,
            )
