"""Tests for the CLI."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from agentic_learn.cli import app, create_agent, handle_command
from agentic_learn.core.agent import Agent
from agentic_learn.core.types import AgentConfig
from agentic_learn.tools import get_tools, list_tools, TOOL_INFO


runner = CliRunner()


# =============================================================================
# Agent Creation Tests
# =============================================================================


class TestCreateAgent:
    """Tests for create_agent function."""

    def test_create_agent_default(self):
        """Test creating agent with defaults."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            agent = create_agent()

        assert agent is not None
        assert isinstance(agent, Agent)
        assert agent.config.model == "claude-sonnet-4-20250514"
        assert agent.config.provider == "anthropic"

    def test_create_agent_custom_model(self):
        """Test creating agent with custom model."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            agent = create_agent(model="custom-model")

        assert agent.config.model == "custom-model"

    def test_create_agent_tier_1(self):
        """Test creating agent with tier 1 tools."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            agent = create_agent(tier=1)

        tools = agent.get_tools()
        tool_names = {t.name for t in tools}

        # Should have core tools
        assert "read" in tool_names
        assert "write" in tool_names
        assert "edit" in tool_names
        assert "bash" in tool_names
        assert "python" in tool_names

        # Should NOT have tier 2+ tools
        assert "gpu" not in tool_names
        assert "jobs" not in tool_names

    def test_create_agent_tier_2(self):
        """Test creating agent with tier 2 tools."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            agent = create_agent(tier=2)

        tools = agent.get_tools()
        tool_names = {t.name for t in tools}

        # Should have core tools
        assert "read" in tool_names

        # Should have tier 2 tools
        assert "gpu" in tool_names
        assert "data" in tool_names
        assert "experiment" in tool_names

        # Should NOT have tier 3 tools
        assert "jobs" not in tool_names

    def test_create_agent_tier_3(self):
        """Test creating agent with tier 3 tools."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            agent = create_agent(tier=3)

        tools = agent.get_tools()
        tool_names = {t.name for t in tools}

        # Should have all tools
        assert "read" in tool_names
        assert "gpu" in tool_names
        assert "jobs" in tool_names
        assert "tune" in tool_names
        assert "viz" in tool_names


# =============================================================================
# Tool Functions Tests
# =============================================================================


class TestToolFunctions:
    """Tests for tool helper functions."""

    def test_get_tools_tier_1(self):
        """Test getting tier 1 tools."""
        tools = get_tools(tier=1)
        names = {t.name for t in tools}

        assert len(tools) == 5
        assert names == {"read", "write", "edit", "bash", "python"}

    def test_get_tools_tier_2(self):
        """Test getting tier 2 tools."""
        tools = get_tools(tier=2)
        names = {t.name for t in tools}

        assert len(tools) == 8
        assert "gpu" in names
        assert "data" in names
        assert "experiment" in names

    def test_get_tools_tier_3(self):
        """Test getting tier 3 tools."""
        tools = get_tools(tier=3)
        names = {t.name for t in tools}

        assert len(tools) == 13
        assert "jobs" in names
        assert "tune" in names
        assert "viz" in names
        assert "notebook" in names
        assert "repro" in names

    def test_get_tools_all(self):
        """Test getting all tools."""
        tools = get_tools(tier="all")
        assert len(tools) == 13

    def test_list_tools(self):
        """Test list_tools output."""
        output = list_tools()

        assert "Tier 1" in output
        assert "Tier 2" in output
        assert "Tier 3" in output
        assert "Core" in output
        assert "read" in output
        assert "gpu" in output
        assert "jobs" in output

    def test_tool_info(self):
        """Test TOOL_INFO dictionary."""
        assert len(TOOL_INFO) == 13

        # Check tier 1
        assert TOOL_INFO["read"]["tier"] == 1
        assert TOOL_INFO["bash"]["tier"] == 1

        # Check tier 2
        assert TOOL_INFO["gpu"]["tier"] == 2
        assert TOOL_INFO["data"]["tier"] == 2

        # Check tier 3
        assert TOOL_INFO["jobs"]["tier"] == 3
        assert TOOL_INFO["tune"]["tier"] == 3


# =============================================================================
# CLI Command Tests
# =============================================================================


class TestCLICommands:
    """Tests for CLI commands."""

    def test_version_flag(self):
        """Test --version flag."""
        result = runner.invoke(app, ["--version"])

        assert result.exit_code == 0
        assert "version" in result.stdout.lower()

    def test_list_tools_flag(self):
        """Test --list-tools flag."""
        result = runner.invoke(app, ["--list-tools"])

        assert result.exit_code == 0
        assert "Tier 1" in result.stdout
        assert "read" in result.stdout

    def test_invalid_tier(self):
        """Test invalid tier value."""
        result = runner.invoke(app, ["--tier", "5", "test"])

        assert result.exit_code == 1
        assert "Invalid tier" in result.stdout

    def test_help(self):
        """Test help output."""
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "DS Coding Agent" in result.stdout
        assert "--tier" in result.stdout


# =============================================================================
# Handle Command Tests
# =============================================================================


class TestHandleCommand:
    """Tests for handle_command function."""

    @pytest.fixture
    def agent(self):
        """Create a mock agent."""
        config = AgentConfig(model="test", provider="anthropic")
        return Agent(config=config)

    async def test_help_command(self, agent):
        """Test /help command."""
        result = await handle_command("/help", agent)
        assert result is True

    async def test_tools_command(self, agent):
        """Test /tools command."""
        result = await handle_command("/tools", agent)
        assert result is True

    async def test_clear_command(self, agent):
        """Test /clear command."""
        from agentic_learn.core.types import Message

        agent.state.messages.append(Message.user("test"))
        assert len(agent.state.messages) == 1

        result = await handle_command("/clear", agent)

        assert result is True
        assert len(agent.state.messages) == 0

    async def test_status_command(self, agent):
        """Test /status command."""
        result = await handle_command("/status", agent)
        assert result is True

    async def test_quit_command(self, agent):
        """Test /quit command."""
        result = await handle_command("/quit", agent)
        assert result is False  # Signals exit

    async def test_exit_command(self, agent):
        """Test /exit command."""
        result = await handle_command("/exit", agent)
        assert result is False  # Signals exit

    async def test_unknown_command(self, agent):
        """Test unknown command."""
        result = await handle_command("/unknown", agent)
        assert result is False


# =============================================================================
# Integration Tests
# =============================================================================


class TestCLIIntegration:
    """Integration tests for CLI."""

    def test_single_message_with_mock(self):
        """Test running single message with mocked agent."""
        # This would require more complex mocking of the entire agent run
        # For now, just verify the CLI structure is correct
        pass
