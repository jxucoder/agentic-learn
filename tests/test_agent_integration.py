"""Integration tests for the agent with mock LLM."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_learn.core.agent import Agent
from agentic_learn.core.tool import Tool, ToolContext, ToolParameter, tool
from agentic_learn.core.types import (
    AgentConfig,
    AgentEvent,
    EventType,
    Message,
    MessageRole,
    ToolCall,
    ToolResult,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config():
    """Create a test config."""
    return AgentConfig(
        model="test-model",
        provider="anthropic",
        api_key="test-key",
    )


@pytest.fixture
def agent(config):
    """Create an agent for testing."""
    return Agent(config=config)


@pytest.fixture
def echo_tool():
    """Create a simple echo tool for testing."""

    @tool(name="echo", description="Echo the input back")
    async def echo(ctx: ToolContext, message: str) -> str:
        return f"Echo: {message}"

    return echo


@pytest.fixture
def error_tool():
    """Create a tool that raises errors."""

    @tool(name="fail", description="Always fails")
    async def fail(ctx: ToolContext) -> str:
        raise ValueError("Intentional failure")

    return fail


@pytest.fixture
def counter_tool():
    """Create a stateful counter tool."""

    class CounterTool(Tool):
        name = "counter"
        description = "Count calls"
        parameters = []

        def __init__(self):
            super().__init__()
            self.count = 0

        async def execute(self, ctx: ToolContext, **kwargs) -> ToolResult:
            self.count += 1
            return ToolResult(tool_call_id="", content=f"Count: {self.count}")

    return CounterTool()


# =============================================================================
# Agent State Tests
# =============================================================================


class TestAgentState:
    """Tests for agent state management."""

    def test_initial_state(self, agent):
        """Test initial agent state."""
        assert agent.state.messages == []
        assert agent.state.is_running is False
        assert agent.state.is_streaming is False
        assert agent.state.error is None

    def test_token_usage_tracking(self, agent):
        """Test token usage initialization."""
        assert agent.state.token_usage["input"] == 0
        assert agent.state.token_usage["output"] == 0


# =============================================================================
# Tool Registration Tests
# =============================================================================


class TestToolRegistration:
    """Tests for tool registration."""

    def test_register_tool(self, agent, echo_tool):
        """Test registering a tool."""
        agent.register_tool(echo_tool)

        assert "echo" in agent._tools
        assert agent.get_tool("echo") is not None

    def test_unregister_tool(self, agent, echo_tool):
        """Test unregistering a tool."""
        agent.register_tool(echo_tool)
        agent.unregister_tool("echo")

        assert "echo" not in agent._tools
        assert agent.get_tool("echo") is None

    def test_get_tools(self, agent, echo_tool, counter_tool):
        """Test getting all tools."""
        agent.register_tool(echo_tool)
        agent.register_tool(counter_tool)

        tools = agent.get_tools()
        assert len(tools) == 2
        names = [t.name for t in tools]
        assert "echo" in names
        assert "counter" in names

    def test_get_tools_schema(self, agent, echo_tool):
        """Test generating tool schemas."""
        agent.register_tool(echo_tool)
        schemas = agent._get_tools_schema()

        assert len(schemas) == 1
        # Anthropic format
        assert schemas[0]["name"] == "echo"


# =============================================================================
# Event Handling Tests
# =============================================================================


class TestEventHandling:
    """Tests for event handling."""

    async def test_event_handler_called(self, agent):
        """Test that event handlers are called."""
        events_received = []

        async def handler(event):
            events_received.append(event)

        agent.on(EventType.AGENT_START, handler)

        await agent._emit(EventType.AGENT_START, {"test": "data"})

        assert len(events_received) == 1
        assert events_received[0].type == EventType.AGENT_START

    async def test_multiple_handlers(self, agent):
        """Test multiple handlers for same event."""
        count = [0]

        async def handler1(event):
            count[0] += 1

        async def handler2(event):
            count[0] += 10

        agent.on(EventType.TURN_START, handler1)
        agent.on(EventType.TURN_START, handler2)

        await agent._emit(EventType.TURN_START)

        assert count[0] == 11


# =============================================================================
# Tool Execution Tests
# =============================================================================


class TestToolExecution:
    """Tests for tool execution."""

    async def test_execute_tool_success(self, agent, echo_tool):
        """Test successful tool execution."""
        agent.register_tool(echo_tool)

        tool_calls = [ToolCall(id="test-1", name="echo", arguments={"message": "hello"})]

        events = []
        async for event in agent._execute_tool_calls(tool_calls):
            events.append(event)

        # Should have start and end events
        event_types = [e.type for e in events]
        assert EventType.TOOL_CALL_START in event_types
        assert EventType.TOOL_CALL_END in event_types

        # Check result was added to messages
        assert len(agent.state.messages) == 1
        assert agent.state.messages[0].role == MessageRole.TOOL
        assert "Echo: hello" in agent.state.messages[0].content

    async def test_execute_unknown_tool(self, agent):
        """Test executing unknown tool."""
        tool_calls = [ToolCall(id="test-1", name="unknown", arguments={})]

        events = []
        async for event in agent._execute_tool_calls(tool_calls):
            events.append(event)

        # Should have error in result
        end_event = next(e for e in events if e.type == EventType.TOOL_CALL_END)
        assert end_event.data["is_error"] is True
        assert "Unknown tool" in end_event.data["result"]

    async def test_execute_tool_error(self, agent, error_tool):
        """Test tool that raises exception."""
        agent.register_tool(error_tool)

        tool_calls = [ToolCall(id="test-1", name="fail", arguments={})]

        events = []
        async for event in agent._execute_tool_calls(tool_calls):
            events.append(event)

        end_event = next(e for e in events if e.type == EventType.TOOL_CALL_END)
        assert end_event.data["is_error"] is True
        assert "Intentional failure" in end_event.data["result"]

    async def test_execute_tool_validation_error(self, agent, echo_tool):
        """Test tool with invalid arguments."""
        agent.register_tool(echo_tool)

        # Missing required 'message' parameter
        tool_calls = [ToolCall(id="test-1", name="echo", arguments={})]

        events = []
        async for event in agent._execute_tool_calls(tool_calls):
            events.append(event)

        end_event = next(e for e in events if e.type == EventType.TOOL_CALL_END)
        assert end_event.data["is_error"] is True

    async def test_multiple_tool_calls(self, agent, counter_tool):
        """Test executing multiple tool calls."""
        agent.register_tool(counter_tool)

        tool_calls = [
            ToolCall(id="test-1", name="counter", arguments={}),
            ToolCall(id="test-2", name="counter", arguments={}),
            ToolCall(id="test-3", name="counter", arguments={}),
        ]

        events = []
        async for event in agent._execute_tool_calls(tool_calls):
            events.append(event)

        # Should have 3 start and 3 end events
        start_events = [e for e in events if e.type == EventType.TOOL_CALL_START]
        end_events = [e for e in events if e.type == EventType.TOOL_CALL_END]
        assert len(start_events) == 3
        assert len(end_events) == 3

        # Counter should have been called 3 times
        assert counter_tool.count == 3


# =============================================================================
# Message Queue Tests
# =============================================================================


class TestMessageQueue:
    """Tests for message queue functionality."""

    async def test_queue_message(self, agent):
        """Test queuing a message."""
        await agent.queue_message("Hello")

        assert not agent._message_queue.empty()
        msg = await agent._message_queue.get()
        assert msg == "Hello"

    async def test_queue_multiple_messages(self, agent):
        """Test queuing multiple messages."""
        await agent.queue_message("First")
        await agent.queue_message("Second")
        await agent.queue_message("Third")

        assert await agent._message_queue.get() == "First"
        assert await agent._message_queue.get() == "Second"
        assert await agent._message_queue.get() == "Third"


# =============================================================================
# Abort Tests
# =============================================================================


class TestAbort:
    """Tests for abort functionality."""

    def test_abort_sets_event(self, agent):
        """Test that abort sets the event."""
        agent.abort()
        assert agent._abort_event.is_set()

    def test_reset_abort(self, agent):
        """Test resetting abort."""
        agent.abort()
        agent._reset_abort()
        assert not agent._abort_event.is_set()

    async def test_abort_stops_tool_execution(self, agent, counter_tool):
        """Test that abort stops tool execution."""
        agent.register_tool(counter_tool)

        # Set abort before execution
        agent.abort()

        tool_calls = [
            ToolCall(id="test-1", name="counter", arguments={}),
            ToolCall(id="test-2", name="counter", arguments={}),
        ]

        events = []
        async for event in agent._execute_tool_calls(tool_calls):
            events.append(event)

        # Only first tool should execute (or none if abort checked at start)
        assert counter_tool.count <= 1


# =============================================================================
# System Prompt Tests
# =============================================================================


class TestSystemPrompt:
    """Tests for system prompt generation."""

    def test_default_system_prompt(self, agent):
        """Test default system prompt."""
        prompt = agent._default_system_prompt()

        assert "ML/DS" in prompt or "assistant" in prompt.lower()

    def test_custom_system_prompt(self, config):
        """Test custom system prompt."""
        config.system_prompt = "You are a custom assistant."
        agent = Agent(config=config)

        prompt = agent._get_system_prompt()

        assert "custom assistant" in prompt

    def test_system_prompt_includes_tools(self, agent, echo_tool):
        """Test that system prompt includes tool descriptions."""
        agent.register_tool(echo_tool)

        prompt = agent._get_system_prompt()

        assert "echo" in prompt.lower()


# =============================================================================
# Mock LLM Integration Tests
# =============================================================================


class TestMockLLMIntegration:
    """Integration tests with mocked LLM responses."""

    @pytest.fixture
    def mock_anthropic_response(self):
        """Create a mock Anthropic response."""

        class MockContentBlock:
            def __init__(self, type, text=None, id=None, name=None, input=None):
                self.type = type
                self.text = text
                self.id = id
                self.name = name
                self.input = input

        class MockUsage:
            input_tokens = 100
            output_tokens = 50

        class MockMessage:
            usage = MockUsage()
            content = [MockContentBlock(type="text", text="Hello from the assistant!")]

        return MockMessage()

    async def test_agent_run_basic(self, agent, echo_tool, mock_anthropic_response):
        """Test basic agent run with mocked LLM."""
        agent.register_tool(echo_tool)

        # Mock the LLM client
        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)
        mock_stream.__aiter__ = lambda self: iter([])  # No streaming events
        mock_stream.get_final_message = AsyncMock(return_value=mock_anthropic_response)

        mock_client = MagicMock()
        mock_client.messages.stream = MagicMock(return_value=mock_stream)

        with patch.object(agent, "_get_llm_client", return_value=mock_client):
            events = []
            async for event in agent.run("Test message"):
                events.append(event)

        event_types = [e.type for e in events]
        assert EventType.AGENT_START in event_types
        assert EventType.AGENT_END in event_types

    async def test_agent_tracks_token_usage(self, agent, mock_anthropic_response):
        """Test that agent tracks token usage."""
        # Create async context manager mock
        mock_stream = MagicMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)
        mock_stream.get_final_message = AsyncMock(return_value=mock_anthropic_response)

        # Make __aiter__ return an async iterator
        async def async_iter():
            return
            yield  # Make it a generator

        mock_stream.__aiter__ = lambda self: async_iter()

        mock_client = MagicMock()
        mock_client.messages.stream = MagicMock(return_value=mock_stream)

        with patch.object(agent, "_get_llm_client", return_value=mock_client):
            async for _ in agent.run("Test"):
                pass

        assert agent.state.token_usage["input"] == 100
        assert agent.state.token_usage["output"] == 50


# =============================================================================
# Message Conversion Tests
# =============================================================================


class TestMessageConversion:
    """Tests for message format conversion."""

    def test_user_message_to_anthropic(self, agent):
        """Test user message conversion."""
        agent.state.messages.append(Message.user("Hello"))

        # The conversion happens in _stream_anthropic, just verify message is correct
        msg = agent.state.messages[0]
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"

    def test_tool_response_message(self, agent):
        """Test tool response message."""
        msg = Message.tool_response("call-1", "echo", "result", is_error=False)

        assert msg.role == MessageRole.TOOL
        assert msg.tool_call_id == "call-1"
        assert msg.name == "echo"
        assert msg.content == "result"

    def test_assistant_message_with_tool_calls(self, agent):
        """Test assistant message with tool calls."""
        tool_calls = [ToolCall(id="1", name="echo", arguments={"message": "hi"})]
        msg = Message.assistant("Let me help", tool_calls=tool_calls)

        assert msg.role == MessageRole.ASSISTANT
        assert msg.content == "Let me help"
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1


# =============================================================================
# Extension Loading Tests
# =============================================================================


class TestExtensionLoading:
    """Tests for extension loading."""

    def test_extension_manager_created(self, agent):
        """Test that extension manager is created."""
        assert agent._extension_manager is not None

    def test_load_extensions_empty_dirs(self, agent):
        """Test loading extensions from empty dirs."""
        agent.load_extensions(directories=[])
        # Should not raise

    def test_load_extensions_nonexistent_dirs(self, agent):
        """Test loading extensions from nonexistent dirs."""
        agent.load_extensions(directories=["/nonexistent/path"])
        # Should not raise, just skip
