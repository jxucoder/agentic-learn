"""Sandboxed execution for safe code running.

Provides isolated execution environments with resource limits:
- Docker-based sandbox (recommended, full isolation)
- Process-based sandbox (fallback, uses resource limits)

Usage:
    sandbox = Sandbox.create()  # Auto-detect best option
    result = await sandbox.run_python("print('hello')")
    result = await sandbox.run_bash("ls -la")
"""

from __future__ import annotations

import asyncio
import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SandboxConfig:
    """Configuration for sandbox execution."""

    # Resource limits
    max_memory_mb: int = 512  # Max memory in MB
    max_cpu_time: int = 30  # Max CPU time in seconds
    max_wall_time: int = 60  # Max wall clock time in seconds
    max_output_size: int = 1_000_000  # Max output in bytes (1MB)
    max_file_size_mb: int = 100  # Max file size in MB

    # Isolation options
    network_enabled: bool = False  # Allow network access
    filesystem_readonly: bool = False  # Mount filesystem read-only
    working_dir: str | None = None  # Working directory (None = temp dir)

    # Docker-specific
    docker_image: str = "python:3.11-slim"  # Docker image for Python
    docker_memory_limit: str = "512m"  # Docker memory limit
    docker_cpu_quota: int = 50000  # Docker CPU quota (50% of one core)


@dataclass
class SandboxResult:
    """Result from sandboxed execution."""

    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False
    memory_exceeded: bool = False
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and not self.memory_exceeded

    @property
    def output(self) -> str:
        """Combined output for convenience."""
        parts = []
        if self.stdout:
            parts.append(self.stdout)
        if self.stderr:
            parts.append(f"stderr:\n{self.stderr}")
        if self.error:
            parts.append(f"Error: {self.error}")
        if self.timed_out:
            parts.append("(execution timed out)")
        if self.memory_exceeded:
            parts.append("(memory limit exceeded)")
        return "\n".join(parts) if parts else "(no output)"


class Sandbox(ABC):
    """Abstract base class for sandbox implementations."""

    def __init__(self, config: SandboxConfig | None = None):
        self.config = config or SandboxConfig()

    @abstractmethod
    async def run_python(self, code: str, timeout: float | None = None) -> SandboxResult:
        """Execute Python code in the sandbox."""
        ...

    @abstractmethod
    async def run_bash(self, command: str, timeout: float | None = None) -> SandboxResult:
        """Execute a bash command in the sandbox."""
        ...

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up sandbox resources."""
        ...

    @classmethod
    def create(cls, config: SandboxConfig | None = None, prefer_docker: bool = True) -> Sandbox:
        """Create the best available sandbox.

        Args:
            config: Sandbox configuration
            prefer_docker: If True, try Docker first

        Returns:
            DockerSandbox if Docker is available, else ProcessSandbox
        """
        if prefer_docker and DockerSandbox.is_available():
            return DockerSandbox(config)
        return ProcessSandbox(config)

    @classmethod
    def is_available(cls) -> bool:
        """Check if this sandbox type is available."""
        return True


class ProcessSandbox(Sandbox):
    """Process-based sandbox using resource limits.

    Uses Unix resource limits (rlimit) for basic isolation.
    Less secure than Docker but works without additional dependencies.
    """

    def __init__(self, config: SandboxConfig | None = None):
        super().__init__(config)
        self._temp_dirs: list[Path] = []

    def _create_temp_dir(self) -> Path:
        """Create a temporary directory for execution."""
        temp_dir = Path(tempfile.mkdtemp(prefix="sandbox_"))
        self._temp_dirs.append(temp_dir)
        return temp_dir

    def _set_resource_limits(self) -> None:
        """Set resource limits for the child process."""
        # Memory limit
        mem_bytes = self.config.max_memory_mb * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        except (ValueError, resource.error):
            pass  # May not be supported

        # CPU time limit
        try:
            resource.setrlimit(
                resource.RLIMIT_CPU,
                (self.config.max_cpu_time, self.config.max_cpu_time),
            )
        except (ValueError, resource.error):
            pass

        # File size limit
        file_bytes = self.config.max_file_size_mb * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_FSIZE, (file_bytes, file_bytes))
        except (ValueError, resource.error):
            pass

        # Number of processes (prevent fork bombs)
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (50, 50))
        except (ValueError, resource.error):
            pass

    async def run_python(self, code: str, timeout: float | None = None) -> SandboxResult:
        """Execute Python code with resource limits."""
        timeout = timeout or self.config.max_wall_time

        # Create temp directory and script
        work_dir = self._create_temp_dir()
        script_path = work_dir / "script.py"
        script_path.write_text(code)

        # Build command
        cmd = [sys.executable, "-u", str(script_path)]

        return await self._run_command(cmd, work_dir, timeout)

    async def run_bash(self, command: str, timeout: float | None = None) -> SandboxResult:
        """Execute bash command with resource limits."""
        timeout = timeout or self.config.max_wall_time

        work_dir = self._create_temp_dir()

        # Build command
        cmd = ["/bin/bash", "-c", command]

        return await self._run_command(cmd, work_dir, timeout)

    async def _run_command(
        self,
        cmd: list[str],
        work_dir: Path,
        timeout: float,
    ) -> SandboxResult:
        """Run a command with resource limits."""

        def preexec():
            """Set up the child process."""
            # Create new process group
            os.setpgrp()
            # Set resource limits
            self._set_resource_limits()

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(work_dir),
                preexec_fn=preexec,
                env=self._get_restricted_env(),
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )

                # Truncate output if too large
                max_size = self.config.max_output_size
                stdout_str = stdout.decode("utf-8", errors="replace")[:max_size]
                stderr_str = stderr.decode("utf-8", errors="replace")[:max_size]

                return SandboxResult(
                    stdout=stdout_str,
                    stderr=stderr_str,
                    exit_code=process.returncode or 0,
                )

            except asyncio.TimeoutError:
                # Kill the entire process group
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                await process.wait()

                return SandboxResult(
                    stdout="",
                    stderr="",
                    exit_code=-1,
                    timed_out=True,
                )

        except MemoryError:
            return SandboxResult(
                stdout="",
                stderr="",
                exit_code=-1,
                memory_exceeded=True,
            )
        except Exception as e:
            return SandboxResult(
                stdout="",
                stderr="",
                exit_code=-1,
                error=str(e),
            )

    def _get_restricted_env(self) -> dict[str, str]:
        """Get a restricted environment for the subprocess."""
        # Start with minimal environment
        env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": "/tmp",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
        }

        # Optionally inherit some variables
        for var in ["PYTHONPATH", "VIRTUAL_ENV"]:
            if var in os.environ:
                env[var] = os.environ[var]

        return env

    async def cleanup(self) -> None:
        """Remove temporary directories."""
        for temp_dir in self._temp_dirs:
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass
        self._temp_dirs.clear()


class DockerSandbox(Sandbox):
    """Docker-based sandbox for full isolation.

    Provides strong isolation through containerization:
    - Separate filesystem namespace
    - Network isolation
    - Resource limits enforced by Docker
    - Clean environment
    """

    def __init__(self, config: SandboxConfig | None = None):
        super().__init__(config)
        self._containers: list[str] = []

    @classmethod
    def is_available(cls) -> bool:
        """Check if Docker is available."""
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    async def run_python(self, code: str, timeout: float | None = None) -> SandboxResult:
        """Execute Python code in a Docker container."""
        timeout = timeout or self.config.max_wall_time

        # Escape code for shell
        escaped_code = code.replace("'", "'\"'\"'")

        cmd = self._build_docker_cmd(
            image=self.config.docker_image,
            command=f"python3 -u -c '{escaped_code}'",
        )

        return await self._run_docker(cmd, timeout)

    async def run_bash(self, command: str, timeout: float | None = None) -> SandboxResult:
        """Execute bash command in a Docker container."""
        timeout = timeout or self.config.max_wall_time

        cmd = self._build_docker_cmd(
            image="alpine:latest",
            command=f"/bin/sh -c '{command}'",
        )

        return await self._run_docker(cmd, timeout)

    def _build_docker_cmd(self, image: str, command: str) -> list[str]:
        """Build the Docker run command."""
        cmd = [
            "docker", "run",
            "--rm",  # Remove container after exit
            "--memory", self.config.docker_memory_limit,
            "--cpu-quota", str(self.config.docker_cpu_quota),
            "--pids-limit", "50",  # Prevent fork bombs
            "--security-opt", "no-new-privileges",
            "--cap-drop", "ALL",  # Drop all capabilities
        ]

        # Network isolation
        if not self.config.network_enabled:
            cmd.extend(["--network", "none"])

        # Read-only filesystem
        if self.config.filesystem_readonly:
            cmd.append("--read-only")
            cmd.extend(["--tmpfs", "/tmp:rw,noexec,nosuid,size=64m"])

        # Add image and command
        cmd.extend([image, "/bin/sh", "-c", command])

        return cmd

    async def _run_docker(self, cmd: list[str], timeout: float) -> SandboxResult:
        """Run a Docker command."""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )

                max_size = self.config.max_output_size
                stdout_str = stdout.decode("utf-8", errors="replace")[:max_size]
                stderr_str = stderr.decode("utf-8", errors="replace")[:max_size]

                # Check for OOM kill (exit code 137)
                memory_exceeded = process.returncode == 137

                return SandboxResult(
                    stdout=stdout_str,
                    stderr=stderr_str,
                    exit_code=process.returncode or 0,
                    memory_exceeded=memory_exceeded,
                )

            except asyncio.TimeoutError:
                # Kill the Docker process
                process.kill()
                await process.wait()

                return SandboxResult(
                    stdout="",
                    stderr="",
                    exit_code=-1,
                    timed_out=True,
                )

        except Exception as e:
            return SandboxResult(
                stdout="",
                stderr="",
                exit_code=-1,
                error=str(e),
            )

    async def cleanup(self) -> None:
        """Clean up any leftover containers."""
        # Docker --rm should handle this, but cleanup just in case
        pass


# Convenience functions


async def run_sandboxed_python(
    code: str,
    config: SandboxConfig | None = None,
    timeout: float | None = None,
) -> SandboxResult:
    """Run Python code in a sandbox (convenience function)."""
    sandbox = Sandbox.create(config)
    try:
        return await sandbox.run_python(code, timeout)
    finally:
        await sandbox.cleanup()


async def run_sandboxed_bash(
    command: str,
    config: SandboxConfig | None = None,
    timeout: float | None = None,
) -> SandboxResult:
    """Run bash command in a sandbox (convenience function)."""
    sandbox = Sandbox.create(config)
    try:
        return await sandbox.run_bash(command, timeout)
    finally:
        await sandbox.cleanup()
