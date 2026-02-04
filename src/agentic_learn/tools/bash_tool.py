"""Bash command execution tool."""

from __future__ import annotations

import asyncio
import os
import shlex
import signal
import subprocess
from pathlib import Path
from typing import Any

from agentic_learn.core.tool import Tool, ToolContext, ToolParameter
from agentic_learn.core.types import ToolResult


class BashTool(Tool):
    """Execute bash commands with optional sandboxing.

    Can run in two modes:
    - Normal: Full access to system (default)
    - Sandboxed: Isolated execution with resource limits
    """

    name = "bash"
    description = """Execute bash commands in the shell.

Use this to:
- Run Python scripts: python train.py
- Install packages: pip install torch
- Git operations: git status, git commit
- File operations: ls, find, grep
- Run tests: pytest tests/
- Build projects: make, npm run build

Features:
- Captures stdout and stderr
- Timeout support (default: 120 seconds)
- Working directory control
- Optional sandboxed mode for untrusted commands

Safety:
- Commands run in the agent's working directory
- Long-running commands will timeout
- Use background=true for commands that don't need output
- Use sandboxed=true for restricted execution

Examples:
- bash command="python train.py --epochs 10"
- bash command="pip install -r requirements.txt"
- bash command="git status"
- bash command="pytest -v" timeout=300"""

    parameters = [
        ToolParameter(
            name="command",
            type=str,
            description="The bash command to execute",
            required=True,
        ),
        ToolParameter(
            name="timeout",
            type=int,
            description="Timeout in seconds (default: 120)",
            required=False,
            default=120,
        ),
        ToolParameter(
            name="cwd",
            type=str,
            description="Working directory for the command (default: current directory)",
            required=False,
            default=None,
        ),
        ToolParameter(
            name="background",
            type=bool,
            description="Run in background without waiting for output (default: false)",
            required=False,
            default=False,
        ),
        ToolParameter(
            name="sandboxed",
            type=bool,
            description="Run in isolated sandbox with resource limits (default: false)",
            required=False,
            default=False,
        ),
    ]

    def __init__(self, sandbox_by_default: bool = False):
        super().__init__()
        self.sandbox_by_default = sandbox_by_default

    async def execute(
        self,
        ctx: ToolContext,
        command: str,
        timeout: int = 120,
        cwd: str | None = None,
        background: bool = False,
        sandboxed: bool = False,
    ) -> ToolResult:
        """Execute a bash command.

        Args:
            ctx: Tool execution context
            command: Bash command to execute
            timeout: Timeout in seconds
            cwd: Working directory (default: current)
            background: Run without waiting for output
            sandboxed: Run in isolated sandbox with resource limits
        """
        # Use sandbox if requested or if default
        use_sandbox = sandboxed or self.sandbox_by_default

        if use_sandbox:
            return await self._execute_sandboxed(command, timeout)

        # Resolve working directory
        working_dir = Path(cwd) if cwd else Path(ctx.cwd)
        if not working_dir.is_absolute():
            working_dir = Path(ctx.cwd) / working_dir

        if not working_dir.exists():
            return ToolResult(
                tool_call_id="",
                content=f"Working directory does not exist: {working_dir}",
                is_error=True,
            )

        # Safety check for dangerous commands
        dangerous = self._check_dangerous(command)
        if dangerous:
            return ToolResult(
                tool_call_id="",
                content=f"Potentially dangerous command blocked: {dangerous}\nIf intended, please confirm.",
                is_error=True,
            )

        try:
            if background:
                return await self._run_background(command, working_dir)
            else:
                return await self._run_foreground(command, working_dir, timeout, ctx)

        except Exception as e:
            return ToolResult(
                tool_call_id="",
                content=f"Error executing command: {str(e)}",
                is_error=True,
            )

    async def _execute_sandboxed(self, command: str, timeout: int) -> ToolResult:
        """Execute command in a sandbox with resource limits."""
        from agentic_learn.core.sandbox import Sandbox, SandboxConfig

        config = SandboxConfig(
            max_wall_time=timeout,
            max_memory_mb=256,
            network_enabled=False,
        )

        sandbox = Sandbox.create(config)
        try:
            result = await sandbox.run_bash(command, timeout=float(timeout))

            return ToolResult(
                tool_call_id="",
                content=result.output,
                is_error=not result.success,
                metadata={
                    "sandboxed": True,
                    "exit_code": result.exit_code,
                    "timed_out": result.timed_out,
                    "memory_exceeded": result.memory_exceeded,
                },
            )
        finally:
            await sandbox.cleanup()

    def _check_dangerous(self, command: str) -> str | None:
        """Check for potentially dangerous commands."""
        dangerous_patterns = [
            ("rm -rf /", "recursive delete of root"),
            ("rm -rf ~", "recursive delete of home"),
            ("rm -rf *", "recursive delete of current directory"),
            (":(){:|:&};:", "fork bomb"),
            ("> /dev/sda", "overwrite disk"),
            ("mkfs.", "format filesystem"),
            ("dd if=", "disk overwrite"),
        ]

        cmd_lower = command.lower()
        for pattern, reason in dangerous_patterns:
            if pattern in cmd_lower:
                return reason

        return None

    async def _run_foreground(
        self,
        command: str,
        cwd: Path,
        timeout: int,
        ctx: ToolContext,
    ) -> ToolResult:
        """Run command in foreground and capture output."""
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd),
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return ToolResult(
                    tool_call_id="",
                    content=f"Command timed out after {timeout} seconds:\n{command}",
                    is_error=True,
                )

            # Check for abort
            if ctx.is_aborted():
                process.kill()
                return ToolResult(
                    tool_call_id="",
                    content="Command aborted",
                    is_error=True,
                )

            # Decode output
            stdout_str = stdout.decode("utf-8", errors="replace")
            stderr_str = stderr.decode("utf-8", errors="replace")

            # Truncate long output
            max_output = 30000
            if len(stdout_str) > max_output:
                stdout_str = stdout_str[:max_output] + "\n... (output truncated)"
            if len(stderr_str) > max_output:
                stderr_str = stderr_str[:max_output] + "\n... (output truncated)"

            # Format result
            output_parts = []

            if stdout_str.strip():
                output_parts.append(stdout_str.strip())

            if stderr_str.strip():
                output_parts.append(f"stderr:\n{stderr_str.strip()}")

            exit_code = process.returncode
            is_error = exit_code != 0

            if is_error:
                output_parts.append(f"\nExit code: {exit_code}")

            content = "\n\n".join(output_parts) if output_parts else "(no output)"

            return ToolResult(
                tool_call_id="",
                content=content,
                is_error=is_error,
                metadata={"exit_code": exit_code},
            )

        except Exception as e:
            return ToolResult(
                tool_call_id="",
                content=f"Failed to execute command: {str(e)}",
                is_error=True,
            )

    async def _run_background(self, command: str, cwd: Path) -> ToolResult:
        """Run command in background."""
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                cwd=str(cwd),
                start_new_session=True,
            )

            return ToolResult(
                tool_call_id="",
                content=f"Started background process:\n  Command: {command}\n  PID: {process.pid}",
                metadata={"pid": process.pid},
            )

        except Exception as e:
            return ToolResult(
                tool_call_id="",
                content=f"Failed to start background process: {str(e)}",
                is_error=True,
            )
