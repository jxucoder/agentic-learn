"""Tests for core agent functionality."""

import pytest

from agentic_learn.core.types import (
    AgentConfig,
    AgentState,
    Message,
    MessageRole,
    ToolCall,
    ToolResult,
)
from agentic_learn.core.tool import Tool, ToolContext, ToolParameter, tool
from agentic_learn.core.agent import Agent


class TestMessage:
    """Tests for Message class."""

    def test_user_message(self):
        msg = Message.user("Hello")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"

    def test_assistant_message(self):
        msg = Message.assistant("Hi there")
        assert msg.role == MessageRole.ASSISTANT
        assert msg.content == "Hi there"
        assert msg.tool_calls is None

    def test_assistant_message_with_tools(self):
        tool_call = ToolCall(id="1", name="test", arguments={"arg": "value"})
        msg = Message.assistant("Let me help", tool_calls=[tool_call])
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "test"

    def test_system_message(self):
        msg = Message.system("You are helpful")
        assert msg.role == MessageRole.SYSTEM

    def test_tool_response(self):
        msg = Message.tool_response("call_1", "my_tool", "result")
        assert msg.role == MessageRole.TOOL
        assert msg.tool_call_id == "call_1"
        assert msg.name == "my_tool"


class TestToolCall:
    """Tests for ToolCall class."""

    def test_tool_call_creation(self):
        tc = ToolCall(id="123", name="python", arguments={"code": "print(1)"})
        assert tc.id == "123"
        assert tc.name == "python"
        assert tc.arguments["code"] == "print(1)"


class TestToolResult:
    """Tests for ToolResult class."""

    def test_tool_result_success(self):
        result = ToolResult(tool_call_id="1", content="Success")
        assert result.is_error is False
        assert result.content == "Success"

    def test_tool_result_error(self):
        result = ToolResult(tool_call_id="1", content="Failed", is_error=True)
        assert result.is_error is True


class TestAgentConfig:
    """Tests for AgentConfig class."""

    def test_default_config(self):
        config = AgentConfig()
        assert config.provider == "anthropic"
        assert config.max_tokens == 8192
        assert config.temperature == 0.0

    def test_custom_config(self):
        config = AgentConfig(
            model="gpt-4",
            provider="openai",
            max_tokens=4096,
        )
        assert config.model == "gpt-4"
        assert config.provider == "openai"
        assert config.max_tokens == 4096


class TestToolDecorator:
    """Tests for the @tool decorator."""

    def test_tool_from_function(self):
        @tool(name="greet", description="Greet someone")
        async def greet(ctx: ToolContext, name: str) -> str:
            return f"Hello, {name}!"

        assert greet.name == "greet"
        assert greet.description == "Greet someone"
        assert len(greet.parameters) == 1
        assert greet.parameters[0].name == "name"

    def test_tool_with_defaults(self):
        @tool(name="add")
        async def add(ctx: ToolContext, a: int, b: int = 10) -> int:
            return a + b

        assert len(add.parameters) == 2
        assert add.parameters[0].required is True
        assert add.parameters[1].required is False
        assert add.parameters[1].default == 10


class TestAgent:
    """Tests for Agent class."""

    def test_agent_creation(self):
        config = AgentConfig()
        agent = Agent(config=config)
        assert agent.config == config
        assert agent.state.messages == []
        assert agent.state.is_running is False

    def test_register_tool(self):
        @tool(name="test_tool")
        async def test_tool(ctx: ToolContext) -> str:
            return "test"

        config = AgentConfig()
        agent = Agent(config=config)
        agent.register_tool(test_tool)

        assert "test_tool" in agent._tools
        assert agent.get_tool("test_tool") is not None

    def test_unregister_tool(self):
        @tool(name="test_tool")
        async def test_tool(ctx: ToolContext) -> str:
            return "test"

        config = AgentConfig()
        agent = Agent(config=config)
        agent.register_tool(test_tool)
        agent.unregister_tool("test_tool")

        assert "test_tool" not in agent._tools


class TestAgentState:
    """Tests for AgentState class."""

    def test_default_state(self):
        state = AgentState()
        assert state.messages == []
        assert state.is_running is False
        assert state.is_streaming is False
        assert state.error is None

    def test_token_usage(self):
        state = AgentState()
        assert state.token_usage["input"] == 0
        assert state.token_usage["output"] == 0
