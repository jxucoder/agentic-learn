"""Tests for sandboxed execution."""

import asyncio
import sys

import pytest

from agentic_learn.core.sandbox import (
    Sandbox,
    SandboxConfig,
    SandboxResult,
    ProcessSandbox,
    DockerSandbox,
)


# =============================================================================
# SandboxConfig Tests
# =============================================================================


class TestSandboxConfig:
    """Tests for SandboxConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SandboxConfig()

        assert config.max_memory_mb == 512
        assert config.max_cpu_time == 30
        assert config.max_wall_time == 60
        assert config.network_enabled is False

    def test_custom_config(self):
        """Test custom configuration."""
        config = SandboxConfig(
            max_memory_mb=1024,
            max_wall_time=120,
            network_enabled=True,
        )

        assert config.max_memory_mb == 1024
        assert config.max_wall_time == 120
        assert config.network_enabled is True


# =============================================================================
# SandboxResult Tests
# =============================================================================


class TestSandboxResult:
    """Tests for SandboxResult."""

    def test_success_result(self):
        """Test successful result."""
        result = SandboxResult(
            stdout="hello",
            stderr="",
            exit_code=0,
        )

        assert result.success is True
        assert "hello" in result.output

    def test_error_result(self):
        """Test error result."""
        result = SandboxResult(
            stdout="",
            stderr="error occurred",
            exit_code=1,
        )

        assert result.success is False
        assert "error" in result.output

    def test_timeout_result(self):
        """Test timeout result."""
        result = SandboxResult(
            stdout="",
            stderr="",
            exit_code=-1,
            timed_out=True,
        )

        assert result.success is False
        assert "timed out" in result.output

    def test_memory_exceeded_result(self):
        """Test memory exceeded result."""
        result = SandboxResult(
            stdout="",
            stderr="",
            exit_code=-1,
            memory_exceeded=True,
        )

        assert result.success is False
        assert "memory" in result.output

    def test_output_combines_parts(self):
        """Test that output combines stdout, stderr, error."""
        result = SandboxResult(
            stdout="out",
            stderr="err",
            exit_code=0,
            error="some error",
        )

        output = result.output
        assert "out" in output
        assert "err" in output
        assert "some error" in output


# =============================================================================
# ProcessSandbox Tests
# =============================================================================


class TestProcessSandbox:
    """Tests for ProcessSandbox."""

    @pytest.fixture
    def sandbox(self):
        """Create a process sandbox."""
        config = SandboxConfig(max_wall_time=10)
        return ProcessSandbox(config)

    async def test_run_python_simple(self, sandbox):
        """Test running simple Python code."""
        result = await sandbox.run_python("print('hello')")

        assert result.success
        assert "hello" in result.stdout

    async def test_run_python_expression(self, sandbox):
        """Test running Python expression."""
        result = await sandbox.run_python("print(2 + 2)")

        assert result.success
        assert "4" in result.stdout

    async def test_run_python_multiline(self, sandbox):
        """Test running multiline Python code."""
        code = """
x = 5
y = 10
print(x + y)
"""
        result = await sandbox.run_python(code)

        assert result.success
        assert "15" in result.stdout

    async def test_run_python_error(self, sandbox):
        """Test Python code that raises error."""
        result = await sandbox.run_python("1/0")

        assert not result.success
        assert "ZeroDivisionError" in result.stderr or "Error" in result.output

    async def test_run_python_syntax_error(self, sandbox):
        """Test Python code with syntax error."""
        result = await sandbox.run_python("def broken(")

        assert not result.success

    async def test_run_python_timeout(self, sandbox):
        """Test Python code timeout."""
        result = await sandbox.run_python(
            "import time; time.sleep(100)",
            timeout=1.0,
        )

        assert not result.success
        assert result.timed_out

    async def test_run_bash_simple(self, sandbox):
        """Test running simple bash command."""
        result = await sandbox.run_bash("echo 'hello'")

        assert result.success
        assert "hello" in result.stdout

    async def test_run_bash_with_args(self, sandbox):
        """Test bash command with arguments."""
        result = await sandbox.run_bash("echo foo bar baz")

        assert result.success
        assert "foo bar baz" in result.stdout

    async def test_run_bash_error(self, sandbox):
        """Test bash command that fails."""
        result = await sandbox.run_bash("ls /nonexistent/path")

        assert not result.success
        assert result.exit_code != 0

    async def test_run_bash_timeout(self, sandbox):
        """Test bash command timeout."""
        result = await sandbox.run_bash("sleep 100", timeout=1.0)

        assert not result.success
        assert result.timed_out

    async def test_cleanup(self, sandbox):
        """Test cleanup removes temp directories."""
        await sandbox.run_python("print('test')")
        assert len(sandbox._temp_dirs) > 0

        await sandbox.cleanup()
        assert len(sandbox._temp_dirs) == 0


# =============================================================================
# DockerSandbox Tests
# =============================================================================


class TestDockerSandbox:
    """Tests for DockerSandbox."""

    @pytest.fixture
    def sandbox(self):
        """Create a Docker sandbox if available."""
        if not DockerSandbox.is_available():
            pytest.skip("Docker not available")
        config = SandboxConfig(max_wall_time=30)
        return DockerSandbox(config)

    def test_is_available(self):
        """Test availability check."""
        # Just verify it returns a boolean
        result = DockerSandbox.is_available()
        assert isinstance(result, bool)

    async def test_run_python_simple(self, sandbox):
        """Test running simple Python in Docker."""
        result = await sandbox.run_python("print('hello from docker')")

        assert result.success
        assert "hello from docker" in result.stdout

    async def test_run_bash_simple(self, sandbox):
        """Test running bash in Docker."""
        result = await sandbox.run_bash("echo 'hello'")

        assert result.success
        assert "hello" in result.stdout


# =============================================================================
# Sandbox Factory Tests
# =============================================================================


class TestSandboxFactory:
    """Tests for Sandbox.create() factory."""

    def test_create_returns_sandbox(self):
        """Test that create returns a Sandbox instance."""
        sandbox = Sandbox.create(prefer_docker=False)
        assert isinstance(sandbox, Sandbox)

    def test_create_process_sandbox(self):
        """Test creating ProcessSandbox explicitly."""
        sandbox = Sandbox.create(prefer_docker=False)
        assert isinstance(sandbox, ProcessSandbox)

    def test_create_with_config(self):
        """Test creating sandbox with config."""
        config = SandboxConfig(max_memory_mb=1024)
        sandbox = Sandbox.create(config, prefer_docker=False)

        assert sandbox.config.max_memory_mb == 1024


# =============================================================================
# Integration Tests
# =============================================================================


class TestSandboxIntegration:
    """Integration tests for sandbox with tools."""

    async def test_python_tool_sandboxed(self):
        """Test PythonTool with sandbox mode."""
        from agentic_learn.tools import PythonTool
        from agentic_learn.core.tool import ToolContext

        tool = PythonTool()
        ctx = ToolContext(cwd="/tmp", agent=None)

        result = await tool.execute(
            ctx,
            code="print('sandboxed!')",
            sandboxed=True,
            timeout=10.0,
        )

        assert not result.is_error
        assert "sandboxed!" in result.content
        assert result.metadata.get("sandboxed") is True

    async def test_bash_tool_sandboxed(self):
        """Test BashTool with sandbox mode."""
        from agentic_learn.tools import BashTool
        from agentic_learn.core.tool import ToolContext

        tool = BashTool()
        ctx = ToolContext(cwd="/tmp", agent=None)

        result = await tool.execute(
            ctx,
            command="echo 'sandboxed!'",
            sandboxed=True,
            timeout=10,
        )

        assert not result.is_error
        assert "sandboxed!" in result.content
        assert result.metadata.get("sandboxed") is True

    async def test_python_tool_sandbox_by_default(self):
        """Test PythonTool with sandbox_by_default."""
        from agentic_learn.tools import PythonTool
        from agentic_learn.core.tool import ToolContext

        tool = PythonTool(sandbox_by_default=True)
        ctx = ToolContext(cwd="/tmp", agent=None)

        result = await tool.execute(ctx, code="print('auto-sandbox')")

        assert result.metadata.get("sandboxed") is True
