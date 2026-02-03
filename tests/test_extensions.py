"""Tests for the extension system."""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agentic_learn.core.agent import Agent
from agentic_learn.core.extension import (
    Command,
    Extension,
    ExtensionAPI,
    ExtensionContext,
    ExtensionManager,
)
from agentic_learn.core.tool import Tool, ToolContext, ToolParameter, tool
from agentic_learn.core.types import AgentConfig, AgentEvent, EventType, ToolResult


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config():
    """Create a test config."""
    return AgentConfig(model="test", provider="anthropic", api_key="test")


@pytest.fixture
def agent(config):
    """Create an agent for testing."""
    return Agent(config=config)


@pytest.fixture
def sample_extension():
    """Create a sample extension for testing."""

    class SampleExtension(Extension):
        name = "sample"
        description = "A sample extension"
        version = "1.0.0"

        def __init__(self):
            super().__init__()
            self.setup_called = False
            self.teardown_called = False

        def setup(self, api: ExtensionAPI) -> None:
            self.setup_called = True

            # Register a tool
            @tool(name="sample_tool", description="A sample tool")
            async def sample_tool(ctx: ToolContext, input: str) -> str:
                return f"Sample: {input}"

            api.register_tool(sample_tool)

            # Register an event handler
            api.on(EventType.TURN_START, self.on_turn_start)

            # Register a command
            api.register_command(
                "sample",
                "Sample command",
                self.sample_command,
            )

        async def on_turn_start(self, ctx: ExtensionContext) -> None:
            pass

        def sample_command(self, ctx: ExtensionContext, args: list[str]) -> str:
            return f"Sample command with args: {args}"

        def teardown(self) -> None:
            self.teardown_called = True

    return SampleExtension()


# =============================================================================
# Extension Tests
# =============================================================================


class TestExtension:
    """Tests for Extension base class."""

    def test_extension_default_name(self):
        """Test default extension name from class name."""

        class MyCustomExtension(Extension):
            pass

        ext = MyCustomExtension()
        assert ext.name == "mycustom"

    def test_extension_explicit_name(self):
        """Test explicit extension name."""

        class MyExtension(Extension):
            name = "explicit_name"

        ext = MyExtension()
        assert ext.name == "explicit_name"

    def test_extension_setup_teardown(self, sample_extension):
        """Test setup and teardown lifecycle."""
        api = MagicMock(spec=ExtensionAPI)

        sample_extension.setup(api)
        assert sample_extension.setup_called

        sample_extension.teardown()
        assert sample_extension.teardown_called


# =============================================================================
# ExtensionAPI Tests
# =============================================================================


class TestExtensionAPI:
    """Tests for ExtensionAPI."""

    @pytest.fixture
    def api(self, agent, sample_extension):
        return ExtensionAPI(sample_extension, agent)

    def test_register_tool(self, api):
        """Test registering a tool."""

        @tool(name="test_tool")
        async def test_tool(ctx: ToolContext) -> str:
            return "test"

        api.register_tool(test_tool)

        tools = api.get_tools()
        assert len(tools) == 1
        assert tools[0].name == "test_tool"

    def test_register_event_handler(self, api):
        """Test registering an event handler."""

        async def handler(ctx: ExtensionContext) -> None:
            pass

        api.on(EventType.AGENT_START, handler)
        api.on(EventType.AGENT_START, handler)  # Can register multiple

        handlers = api.get_event_handlers(EventType.AGENT_START)
        assert len(handlers) == 2

    def test_register_command(self, api):
        """Test registering a command."""

        def handler(ctx: ExtensionContext, args: list[str]) -> str:
            return "result"

        api.register_command("test", "Test command", handler)

        commands = api.get_commands()
        assert "test" in commands
        assert commands["test"].description == "Test command"

    def test_get_nonexistent_event_handlers(self, api):
        """Test getting handlers for unregistered event type."""
        handlers = api.get_event_handlers(EventType.ERROR)
        assert handlers == []


# =============================================================================
# ExtensionContext Tests
# =============================================================================


class TestExtensionContext:
    """Tests for ExtensionContext."""

    @pytest.fixture
    def ctx(self, agent):
        return ExtensionContext(agent=agent, cwd="/test")

    def test_is_idle(self, ctx, agent):
        """Test is_idle check."""
        assert ctx.is_idle() is True

        agent.state.is_running = True
        assert ctx.is_idle() is False

    def test_abort(self, ctx, agent):
        """Test abort functionality."""
        ctx.abort()
        assert agent._abort_event.is_set()

    def test_get_messages(self, ctx, agent):
        """Test getting messages."""
        from agentic_learn.core.types import Message

        agent.state.messages.append(Message.user("Hello"))

        messages = ctx.get_messages()
        assert len(messages) == 1
        assert messages[0].content == "Hello"

        # Should be a copy
        messages.clear()
        assert len(ctx.get_messages()) == 1

    def test_get_token_usage(self, ctx, agent):
        """Test getting token usage."""
        agent.state.token_usage["input"] = 100

        usage = ctx.get_token_usage()
        assert usage["input"] == 100

        # Should be a copy
        usage["input"] = 999
        assert ctx.get_token_usage()["input"] == 100

    async def test_send_message(self, ctx, agent):
        """Test sending a message."""
        await ctx.send_message("Test message")

        msg = await agent._message_queue.get()
        assert msg == "Test message"


# =============================================================================
# ExtensionManager Tests
# =============================================================================


class TestExtensionManager:
    """Tests for ExtensionManager."""

    @pytest.fixture
    def manager(self, agent):
        return ExtensionManager(agent=agent)

    def test_load_extension(self, manager, sample_extension):
        """Test loading an extension."""
        manager.load_extension(sample_extension)

        assert "sample" in manager.extensions
        assert sample_extension.setup_called

    def test_loaded_extension_tools_registered(self, manager, sample_extension):
        """Test that tools from extension are registered with agent."""
        manager.load_extension(sample_extension)

        # Tool should be registered with agent
        assert manager.agent.get_tool("sample_tool") is not None

    def test_unload_extension(self, manager, sample_extension):
        """Test unloading an extension."""
        manager.load_extension(sample_extension)
        manager.unload_extension("sample")

        assert "sample" not in manager.extensions
        assert sample_extension.teardown_called

    def test_get_all_tools(self, manager, sample_extension):
        """Test getting all tools from extensions."""
        manager.load_extension(sample_extension)

        tools = manager.get_all_tools()
        assert len(tools) == 1
        assert tools[0].name == "sample_tool"

    def test_get_all_commands(self, manager, sample_extension):
        """Test getting all commands from extensions."""
        manager.load_extension(sample_extension)

        commands = manager.get_all_commands()
        assert "sample" in commands

    async def test_emit_event(self, manager, agent):
        """Test emitting events to extensions."""
        events_received = []

        class EventExtension(Extension):
            name = "events"

            def setup(self, api: ExtensionAPI) -> None:
                api.on(EventType.TURN_START, self.on_event)

            async def on_event(self, ctx: ExtensionContext) -> None:
                events_received.append(ctx.event)

        manager.load_extension(EventExtension())

        event = AgentEvent(type=EventType.TURN_START, data={"test": "data"})
        await manager.emit_event(event)

        assert len(events_received) == 1
        assert events_received[0].type == EventType.TURN_START

    def test_load_from_directory_nonexistent(self, manager):
        """Test loading from nonexistent directory."""
        manager.load_from_directory(Path("/nonexistent"))
        # Should not raise, just skip
        assert len(manager.extensions) == 0


# =============================================================================
# File-based Extension Loading Tests
# =============================================================================


class TestFileExtensionLoading:
    """Tests for loading extensions from files."""

    @pytest.fixture
    def manager(self, agent):
        return ExtensionManager(agent=agent)

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for extension files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_load_from_path_class_style(self, manager, temp_dir):
        """Test loading extension with class style."""
        ext_file = temp_dir / "my_ext.py"
        ext_file.write_text("""
from agentic_learn.core.extension import Extension, ExtensionAPI

class MyExtension(Extension):
    name = "my_ext"
    description = "Test extension"

    def setup(self, api: ExtensionAPI) -> None:
        pass
""")

        manager.load_from_path(ext_file)
        assert "my_ext" in manager.extensions

    def test_load_from_path_functional_style(self, manager, temp_dir):
        """Test loading extension with functional style."""
        ext_file = temp_dir / "func_ext.py"
        ext_file.write_text("""
def setup(api):
    pass
""")

        manager.load_from_path(ext_file)
        assert "func_ext" in manager.extensions

    def test_load_from_path_factory_style(self, manager, temp_dir):
        """Test loading extension with factory function."""
        ext_file = temp_dir / "factory_ext.py"
        ext_file.write_text("""
from agentic_learn.core.extension import Extension, ExtensionAPI

class MyExtension(Extension):
    name = "factory"

    def setup(self, api: ExtensionAPI) -> None:
        pass

extension = MyExtension()
""")

        manager.load_from_path(ext_file)
        assert "factory" in manager.extensions

    def test_load_from_directory(self, manager, temp_dir):
        """Test loading multiple extensions from directory."""
        # Create multiple extension files
        (temp_dir / "ext1.py").write_text("""
from agentic_learn.core.extension import Extension

class Ext1(Extension):
    name = "ext1"
""")
        (temp_dir / "ext2.py").write_text("""
from agentic_learn.core.extension import Extension

class Ext2(Extension):
    name = "ext2"
""")
        # Skip files starting with underscore
        (temp_dir / "_private.py").write_text("# should be skipped")

        manager.load_from_directory(temp_dir)

        assert "ext1" in manager.extensions
        assert "ext2" in manager.extensions
        assert "_private" not in manager.extensions

    def test_load_from_path_not_found(self, manager):
        """Test loading from nonexistent path."""
        with pytest.raises(FileNotFoundError):
            manager.load_from_path(Path("/nonexistent/extension.py"))


# =============================================================================
# Command Tests
# =============================================================================


class TestCommand:
    """Tests for Command class."""

    def test_command_creation(self):
        """Test creating a command."""

        def handler(ctx: ExtensionContext, args: list[str]) -> str:
            return "result"

        cmd = Command(name="test", description="A test command", handler=handler)

        assert cmd.name == "test"
        assert cmd.description == "A test command"

    def test_command_handler_execution(self):
        """Test executing command handler."""

        def handler(ctx: ExtensionContext, args: list[str]) -> str:
            return f"Args: {', '.join(args)}"

        cmd = Command(name="test", description="Test", handler=handler)

        result = cmd.handler(None, ["arg1", "arg2"])
        assert result == "Args: arg1, arg2"
