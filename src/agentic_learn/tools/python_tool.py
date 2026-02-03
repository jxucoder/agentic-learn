"""Python code execution tool."""

from __future__ import annotations

import asyncio
import io
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

from agentic_learn.core.tool import Tool, ToolContext, ToolParameter
from agentic_learn.core.types import ToolResult


class PythonTool(Tool):
    """Execute Python code in a sandboxed environment.

    Supports data science workflows with access to common libraries.
    Maintains state across executions within the same session.
    """

    name = "python"
    description = """Execute Python code for data analysis, ML, and general computation.

Features:
- Persistent namespace: variables defined in one call are available in subsequent calls
- Common libraries pre-imported: numpy, pandas, matplotlib, sklearn (if available)
- Captures stdout, stderr, and the last expression value
- Handles async code automatically

Use this for:
- Data exploration and manipulation
- Model training and evaluation
- Visualization (figures are saved to files)
- General computation and scripting

Note: Long-running operations may timeout. Break large tasks into smaller steps."""

    parameters = [
        ToolParameter(
            name="code",
            type=str,
            description="Python code to execute. Can be multiple lines.",
            required=True,
        ),
        ToolParameter(
            name="timeout",
            type=float,
            description="Maximum execution time in seconds (default: 60)",
            required=False,
            default=60.0,
        ),
    ]

    def __init__(self):
        super().__init__()
        # Persistent namespace for code execution
        self._namespace: dict[str, Any] = {}
        self._setup_namespace()

    def _setup_namespace(self) -> None:
        """Set up the execution namespace with common imports."""
        # Pre-populate with common imports (lazy - only if available)
        setup_code = """
import sys
import os
import json
import math
import random
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from typing import Any, Dict, List, Optional, Tuple

# Try to import common DS libraries
try:
    import numpy as np
except ImportError:
    np = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    matplotlib = None

try:
    import sklearn
except ImportError:
    sklearn = None

try:
    import torch
except ImportError:
    torch = None

try:
    import tensorflow as tf
except ImportError:
    tf = None
"""
        try:
            exec(setup_code, self._namespace)
        except Exception:
            pass  # Some imports may fail, that's ok

    def reset_namespace(self) -> None:
        """Reset the execution namespace to initial state."""
        self._namespace.clear()
        self._setup_namespace()

    async def execute(
        self,
        ctx: ToolContext,
        code: str,
        timeout: float = 60.0,
    ) -> ToolResult:
        """Execute Python code and return the result."""
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        result_value = None
        error_occurred = False
        error_message = ""

        # Update namespace with context info
        self._namespace["__cwd__"] = ctx.cwd
        self._namespace["__agent__"] = ctx.agent

        try:
            # Compile and execute in a thread to support timeout
            def run_code():
                nonlocal result_value, error_occurred, error_message

                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    try:
                        # Try to compile as expression first (to get return value)
                        try:
                            compiled = compile(code, "<agent>", "eval")
                            result_value = eval(compiled, self._namespace)
                        except SyntaxError:
                            # Not an expression, execute as statements
                            compiled = compile(code, "<agent>", "exec")
                            exec(compiled, self._namespace)

                            # Check if last line could be an expression
                            lines = code.strip().split("\n")
                            if lines:
                                last_line = lines[-1].strip()
                                if last_line and not any(
                                    last_line.startswith(kw)
                                    for kw in (
                                        "import", "from", "def", "class", "if", "for",
                                        "while", "try", "with", "return", "raise", "assert",
                                        "#", "pass", "break", "continue"
                                    )
                                ) and "=" not in last_line.split("#")[0]:
                                    try:
                                        result_value = eval(last_line, self._namespace)
                                    except Exception:
                                        pass
                    except Exception as e:
                        error_occurred = True
                        error_message = traceback.format_exc()

            # Run with timeout
            loop = asyncio.get_event_loop()
            await asyncio.wait_for(
                loop.run_in_executor(None, run_code),
                timeout=timeout,
            )

        except asyncio.TimeoutError:
            error_occurred = True
            error_message = f"Execution timed out after {timeout} seconds"

        # Collect output
        stdout_output = stdout_capture.getvalue()
        stderr_output = stderr_capture.getvalue()

        # Format result
        output_parts = []

        if stdout_output:
            output_parts.append(f"stdout:\n{stdout_output}")

        if stderr_output:
            output_parts.append(f"stderr:\n{stderr_output}")

        if error_occurred:
            output_parts.append(f"Error:\n{error_message}")
        elif result_value is not None:
            # Format the result value
            try:
                if hasattr(result_value, "_repr_html_"):
                    # DataFrames and similar
                    result_str = repr(result_value)
                    if len(result_str) > 2000:
                        result_str = result_str[:2000] + "\n... (truncated)"
                else:
                    result_str = repr(result_value)
                    if len(result_str) > 5000:
                        result_str = result_str[:5000] + "\n... (truncated)"
                output_parts.append(f"Result:\n{result_str}")
            except Exception as e:
                output_parts.append(f"Result: <unable to represent: {e}>")

        if not output_parts:
            output_parts.append("(no output)")

        return ToolResult(
            tool_call_id="",
            content="\n\n".join(output_parts),
            is_error=error_occurred,
            metadata={
                "has_stdout": bool(stdout_output),
                "has_stderr": bool(stderr_output),
                "has_result": result_value is not None,
            },
        )
