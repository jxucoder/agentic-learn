"""Tests for built-in tools."""

import asyncio
import os
import tempfile
from pathlib import Path

import pytest

from agentic_learn.core.tool import ToolContext
from agentic_learn.tools import (
    ReadTool,
    WriteTool,
    EditTool,
    BashTool,
    PythonTool,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def ctx(temp_dir):
    """Create a ToolContext for testing."""
    return ToolContext(cwd=str(temp_dir), agent=None)


@pytest.fixture
def sample_file(temp_dir):
    """Create a sample file for testing."""
    filepath = temp_dir / "sample.py"
    content = """def hello():
    print("Hello, World!")

def add(a, b):
    return a + b

if __name__ == "__main__":
    hello()
"""
    filepath.write_text(content)
    return filepath


# =============================================================================
# ReadTool Tests
# =============================================================================


class TestReadTool:
    """Tests for ReadTool."""

    @pytest.fixture
    def read_tool(self):
        return ReadTool()

    async def test_read_file(self, read_tool, ctx, sample_file):
        """Test reading a file."""
        result = await read_tool.execute(ctx, path=str(sample_file))

        assert not result.is_error
        assert "def hello():" in result.content
        assert "def add(a, b):" in result.content
        assert "8 lines" in result.content or "lines" in result.content.lower()

    async def test_read_file_relative_path(self, read_tool, ctx, sample_file):
        """Test reading a file with relative path."""
        result = await read_tool.execute(ctx, path="sample.py")

        assert not result.is_error
        assert "def hello():" in result.content

    async def test_read_file_not_found(self, read_tool, ctx):
        """Test reading a non-existent file."""
        result = await read_tool.execute(ctx, path="nonexistent.py")

        assert result.is_error
        assert "not found" in result.content.lower()

    async def test_read_file_with_offset_and_limit(self, read_tool, ctx, sample_file):
        """Test reading with offset and limit."""
        result = await read_tool.execute(ctx, path=str(sample_file), offset=3, limit=2)

        assert not result.is_error
        # Should show lines 3-4
        assert "Showing lines 3-4" in result.content or "3" in result.content

    async def test_read_directory(self, read_tool, ctx, temp_dir):
        """Test listing a directory."""
        # Create some files
        (temp_dir / "file1.py").write_text("content1")
        (temp_dir / "file2.py").write_text("content2")
        (temp_dir / "subdir").mkdir()

        result = await read_tool.execute(ctx, path=str(temp_dir))

        assert not result.is_error
        assert "file1.py" in result.content
        assert "file2.py" in result.content
        assert "subdir" in result.content

    async def test_read_binary_file(self, read_tool, ctx, temp_dir):
        """Test reading a binary file."""
        binary_file = temp_dir / "test.png"
        binary_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        result = await read_tool.execute(ctx, path=str(binary_file))

        assert not result.is_error
        assert "Binary file" in result.content or "image" in result.content.lower()


# =============================================================================
# WriteTool Tests
# =============================================================================


class TestWriteTool:
    """Tests for WriteTool."""

    @pytest.fixture
    def write_tool(self):
        return WriteTool()

    async def test_write_new_file(self, write_tool, ctx, temp_dir):
        """Test writing a new file."""
        filepath = temp_dir / "new_file.py"
        content = "print('Hello')"

        result = await write_tool.execute(ctx, path=str(filepath), content=content)

        assert not result.is_error
        assert "Created" in result.content
        assert filepath.exists()
        assert filepath.read_text() == content

    async def test_write_creates_directories(self, write_tool, ctx, temp_dir):
        """Test that write creates parent directories."""
        filepath = temp_dir / "subdir" / "nested" / "file.py"
        content = "# nested file"

        result = await write_tool.execute(ctx, path=str(filepath), content=content)

        assert not result.is_error
        assert filepath.exists()
        assert filepath.read_text() == content

    async def test_write_overwrite_existing(self, write_tool, ctx, sample_file):
        """Test overwriting an existing file."""
        new_content = "# replaced content"

        result = await write_tool.execute(
            ctx, path=str(sample_file), content=new_content, overwrite=True
        )

        assert not result.is_error
        assert "Overwrote" in result.content
        assert sample_file.read_text() == new_content

    async def test_write_no_overwrite_error(self, write_tool, ctx, sample_file):
        """Test error when overwrite=False and file exists."""
        result = await write_tool.execute(
            ctx, path=str(sample_file), content="new", overwrite=False
        )

        assert result.is_error
        assert "already exists" in result.content.lower()

    async def test_write_relative_path(self, write_tool, ctx, temp_dir):
        """Test writing with relative path."""
        result = await write_tool.execute(ctx, path="relative.txt", content="test")

        assert not result.is_error
        assert (temp_dir / "relative.txt").exists()


# =============================================================================
# EditTool Tests
# =============================================================================


class TestEditTool:
    """Tests for EditTool."""

    @pytest.fixture
    def edit_tool(self):
        return EditTool()

    async def test_edit_single_replacement(self, edit_tool, ctx, sample_file):
        """Test single text replacement."""
        result = await edit_tool.execute(
            ctx,
            path=str(sample_file),
            old_string='print("Hello, World!")',
            new_string='print("Goodbye, World!")',
        )

        assert not result.is_error
        assert "Edited" in result.content
        assert 'print("Goodbye, World!")' in sample_file.read_text()

    async def test_edit_replace_all(self, edit_tool, ctx, temp_dir):
        """Test replace_all flag."""
        filepath = temp_dir / "multi.py"
        filepath.write_text("foo\nfoo\nfoo")

        result = await edit_tool.execute(
            ctx,
            path=str(filepath),
            old_string="foo",
            new_string="bar",
            replace_all=True,
        )

        assert not result.is_error
        assert filepath.read_text() == "bar\nbar\nbar"

    async def test_edit_multiple_matches_error(self, edit_tool, ctx, temp_dir):
        """Test error when multiple matches without replace_all."""
        filepath = temp_dir / "multi.py"
        filepath.write_text("foo\nfoo\nfoo")

        result = await edit_tool.execute(
            ctx,
            path=str(filepath),
            old_string="foo",
            new_string="bar",
        )

        assert result.is_error
        assert "Multiple occurrences" in result.content or "multiple" in result.content.lower()

    async def test_edit_text_not_found(self, edit_tool, ctx, sample_file):
        """Test error when text not found."""
        result = await edit_tool.execute(
            ctx,
            path=str(sample_file),
            old_string="nonexistent text",
            new_string="replacement",
        )

        assert result.is_error
        assert "not found" in result.content.lower()

    async def test_edit_file_not_found(self, edit_tool, ctx):
        """Test error when file not found."""
        result = await edit_tool.execute(
            ctx,
            path="nonexistent.py",
            old_string="foo",
            new_string="bar",
        )

        assert result.is_error
        assert "not found" in result.content.lower()

    async def test_edit_multiline(self, edit_tool, ctx, temp_dir):
        """Test multiline replacement."""
        filepath = temp_dir / "test.py"
        original = """def old():
    pass"""
        replacement = """def new():
    return True"""
        filepath.write_text(original)

        result = await edit_tool.execute(
            ctx,
            path=str(filepath),
            old_string=original,
            new_string=replacement,
        )

        assert not result.is_error
        assert filepath.read_text() == replacement


# =============================================================================
# BashTool Tests
# =============================================================================


class TestBashTool:
    """Tests for BashTool."""

    @pytest.fixture
    def bash_tool(self):
        return BashTool()

    async def test_bash_simple_command(self, bash_tool, ctx):
        """Test running a simple command."""
        result = await bash_tool.execute(ctx, command="echo 'hello world'")

        assert not result.is_error
        assert "hello world" in result.content

    async def test_bash_command_with_output(self, bash_tool, ctx, temp_dir):
        """Test command with file output."""
        # Create a file
        (temp_dir / "test.txt").write_text("line1\nline2\nline3")

        result = await bash_tool.execute(ctx, command="cat test.txt")

        assert not result.is_error
        assert "line1" in result.content
        assert "line2" in result.content

    async def test_bash_command_error(self, bash_tool, ctx):
        """Test command that returns error."""
        result = await bash_tool.execute(ctx, command="ls /nonexistent/path")

        assert result.is_error
        assert result.metadata.get("exit_code") != 0

    async def test_bash_working_directory(self, bash_tool, ctx, temp_dir):
        """Test command with custom working directory."""
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        (subdir / "file.txt").write_text("content")

        result = await bash_tool.execute(ctx, command="ls", cwd=str(subdir))

        assert not result.is_error
        assert "file.txt" in result.content

    async def test_bash_timeout(self, bash_tool, ctx):
        """Test command timeout."""
        result = await bash_tool.execute(ctx, command="sleep 10", timeout=1)

        assert result.is_error
        assert "timed out" in result.content.lower()

    async def test_bash_dangerous_command_blocked(self, bash_tool, ctx):
        """Test that dangerous commands are blocked."""
        result = await bash_tool.execute(ctx, command="rm -rf /")

        assert result.is_error
        assert "dangerous" in result.content.lower() or "blocked" in result.content.lower()

    async def test_bash_background(self, bash_tool, ctx):
        """Test running command in background."""
        result = await bash_tool.execute(ctx, command="echo 'bg'", background=True)

        assert not result.is_error
        assert "background" in result.content.lower() or "PID" in result.content


# =============================================================================
# PythonTool Tests
# =============================================================================


class TestPythonTool:
    """Tests for PythonTool."""

    @pytest.fixture
    def python_tool(self):
        return PythonTool()

    async def test_python_simple_expression(self, python_tool, ctx):
        """Test evaluating a simple expression."""
        result = await python_tool.execute(ctx, code="2 + 2")

        assert not result.is_error
        assert "4" in result.content

    async def test_python_print_output(self, python_tool, ctx):
        """Test capturing print output."""
        result = await python_tool.execute(ctx, code='print("hello")')

        assert not result.is_error
        assert "hello" in result.content

    async def test_python_variable_persistence(self, python_tool, ctx):
        """Test that variables persist across executions."""
        # First execution
        await python_tool.execute(ctx, code="x = 42")

        # Second execution should have access to x
        result = await python_tool.execute(ctx, code="x * 2")

        assert not result.is_error
        assert "84" in result.content

    async def test_python_error_handling(self, python_tool, ctx):
        """Test error handling."""
        result = await python_tool.execute(ctx, code="1/0")

        assert result.is_error
        assert "ZeroDivisionError" in result.content

    async def test_python_syntax_error(self, python_tool, ctx):
        """Test syntax error handling."""
        result = await python_tool.execute(ctx, code="def broken(")

        assert result.is_error
        assert "SyntaxError" in result.content

    async def test_python_multiline_code(self, python_tool, ctx):
        """Test multiline code execution."""
        code = """
def greet(name):
    return f"Hello, {name}!"

result = greet("World")
print(result)
"""
        result = await python_tool.execute(ctx, code=code)

        assert not result.is_error
        assert "Hello, World!" in result.content

    async def test_python_imports(self, python_tool, ctx):
        """Test that basic imports work."""
        # Use print to ensure output is captured
        result = await python_tool.execute(ctx, code="import json; print(json.dumps({'a': 1}))")

        assert not result.is_error
        assert '{"a": 1}' in result.content

    async def test_python_timeout(self, python_tool, ctx):
        """Test execution timeout."""
        result = await python_tool.execute(
            ctx,
            code="import time; time.sleep(10)",
            timeout=1.0,
        )

        assert result.is_error
        assert "timed out" in result.content.lower()

    async def test_python_reset_namespace(self, python_tool, ctx):
        """Test resetting namespace."""
        await python_tool.execute(ctx, code="x = 123")
        python_tool.reset_namespace()

        result = await python_tool.execute(ctx, code="x")

        assert result.is_error
        assert "NameError" in result.content


# =============================================================================
# Tool Definition Tests
# =============================================================================


class TestToolDefinitions:
    """Tests for tool schema generation."""

    def test_read_tool_openai_schema(self):
        """Test ReadTool OpenAI schema."""
        tool = ReadTool()
        schema = tool.get_definition().to_openai_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "read"
        assert "path" in schema["function"]["parameters"]["properties"]
        assert "path" in schema["function"]["parameters"]["required"]

    def test_write_tool_anthropic_schema(self):
        """Test WriteTool Anthropic schema."""
        tool = WriteTool()
        schema = tool.get_definition().to_anthropic_schema()

        assert schema["name"] == "write"
        assert "path" in schema["input_schema"]["properties"]
        assert "content" in schema["input_schema"]["properties"]
        assert "path" in schema["input_schema"]["required"]
        assert "content" in schema["input_schema"]["required"]

    def test_tool_validate_args_required(self):
        """Test argument validation for required params."""
        tool = ReadTool()

        # Missing required path
        is_valid, error = tool.validate_args({})
        assert not is_valid
        assert "path" in error

        # With path
        is_valid, error = tool.validate_args({"path": "test.py"})
        assert is_valid

    def test_tool_validate_args_type_coercion(self):
        """Test argument type coercion."""
        tool = ReadTool()

        # offset as string should be coerced to int
        args = {"path": "test.py", "offset": "5"}
        is_valid, error = tool.validate_args(args)
        assert is_valid
        assert args["offset"] == 5
