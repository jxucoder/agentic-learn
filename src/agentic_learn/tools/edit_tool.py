"""File editing tool."""

from __future__ import annotations

import difflib
import os
from pathlib import Path
from typing import Any

from agentic_learn.core.tool import Tool, ToolContext, ToolParameter
from agentic_learn.core.types import ToolResult


class EditTool(Tool):
    """Edit files with find-and-replace operations."""

    name = "edit"
    description = """Edit a file by replacing specific text.

Operations:
- Find and replace exact text
- Replace all occurrences with replace_all=true
- Must read the file first to know what to replace

How it works:
1. Searches for 'old_string' in the file
2. Replaces it with 'new_string'
3. Saves the modified file

Requirements:
- old_string must exist in the file (exact match)
- old_string must be unique unless using replace_all
- Include enough context in old_string to make it unique

Examples:
- edit path="model.py" old_string="lr = 0.001" new_string="lr = 0.0001"
- edit path="config.py" old_string="DEBUG = True" new_string="DEBUG = False"
- edit path="utils.py" old_string="print(" new_string="logger.info(" replace_all=true"""

    parameters = [
        ToolParameter(
            name="path",
            type=str,
            description="Path to the file to edit",
            required=True,
        ),
        ToolParameter(
            name="old_string",
            type=str,
            description="The exact text to find and replace",
            required=True,
        ),
        ToolParameter(
            name="new_string",
            type=str,
            description="The text to replace it with",
            required=True,
        ),
        ToolParameter(
            name="replace_all",
            type=bool,
            description="Replace all occurrences (default: false, requires unique match)",
            required=False,
            default=False,
        ),
    ]

    async def execute(
        self,
        ctx: ToolContext,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> ToolResult:
        """Edit a file by replacing text."""
        # Resolve path
        path_obj = Path(path)
        if not path_obj.is_absolute():
            path_obj = Path(ctx.cwd) / path_obj

        if not path_obj.exists():
            return ToolResult(
                tool_call_id="",
                content=f"File not found: {path}",
                is_error=True,
            )

        if path_obj.is_dir():
            return ToolResult(
                tool_call_id="",
                content=f"Path is a directory: {path}",
                is_error=True,
            )

        try:
            # Read the file
            with open(path_obj, "r", encoding="utf-8") as f:
                content = f.read()

            # Check if old_string exists
            count = content.count(old_string)

            if count == 0:
                # Try to find similar text to help debug
                lines = content.split("\n")
                similar = []
                for i, line in enumerate(lines, 1):
                    if any(word in line for word in old_string.split()[:3]):
                        similar.append(f"  Line {i}: {line[:100]}")

                hint = ""
                if similar:
                    hint = "\n\nSimilar lines found:\n" + "\n".join(similar[:5])

                return ToolResult(
                    tool_call_id="",
                    content=f"Text not found in file: {path}\n\nSearched for:\n{old_string[:200]}{hint}",
                    is_error=True,
                )

            if count > 1 and not replace_all:
                # Show occurrences to help user provide more context
                lines = content.split("\n")
                occurrences = []
                for i, line in enumerate(lines, 1):
                    if old_string in line or (len(old_string.split("\n")) > 1 and old_string.split("\n")[0] in line):
                        occurrences.append(f"  Line {i}: {line[:100]}")

                return ToolResult(
                    tool_call_id="",
                    content=f"Multiple occurrences ({count}) found. Use replace_all=true or provide more context.\n\nOccurrences:\n" + "\n".join(occurrences[:10]),
                    is_error=True,
                )

            # Perform replacement
            if replace_all:
                new_content = content.replace(old_string, new_string)
            else:
                new_content = content.replace(old_string, new_string, 1)

            # Write back
            with open(path_obj, "w", encoding="utf-8") as f:
                f.write(new_content)

            # Generate diff for feedback
            diff = self._generate_diff(content, new_content, path)

            return ToolResult(
                tool_call_id="",
                content=f"Edited: {path}\nReplacements: {count if replace_all else 1}\n\n{diff}",
                metadata={"replacements": count if replace_all else 1},
            )

        except UnicodeDecodeError:
            return ToolResult(
                tool_call_id="",
                content=f"Cannot read file as text: {path}",
                is_error=True,
            )
        except PermissionError:
            return ToolResult(
                tool_call_id="",
                content=f"Permission denied: {path}",
                is_error=True,
            )

    def _generate_diff(self, old_content: str, new_content: str, path: str) -> str:
        """Generate a unified diff between old and new content."""
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)

        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            lineterm="",
        )

        diff_lines = list(diff)

        # Truncate if too long
        if len(diff_lines) > 50:
            diff_lines = diff_lines[:50] + ["\n... (diff truncated)"]

        return "".join(diff_lines)
