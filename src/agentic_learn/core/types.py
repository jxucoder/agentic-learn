"""Core types for the agent system."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Role of a message in the conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(BaseModel):
    """A tool call requested by the assistant."""

    id: str
    name: str
    arguments: dict[str, Any]


class ToolResult(BaseModel):
    """Result of a tool execution."""

    tool_call_id: str
    content: str | list[dict[str, Any]]  # text or structured content (images, etc.)
    is_error: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    """A message in the conversation."""

    role: MessageRole
    content: str | list[dict[str, Any]] | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None  # For tool response messages
    name: str | None = None  # Tool name for tool responses
    timestamp: datetime = Field(default_factory=datetime.now)

    @classmethod
    def user(cls, content: str) -> Message:
        """Create a user message."""
        return cls(role=MessageRole.USER, content=content)

    @classmethod
    def assistant(cls, content: str, tool_calls: list[ToolCall] | None = None) -> Message:
        """Create an assistant message."""
        return cls(role=MessageRole.ASSISTANT, content=content, tool_calls=tool_calls)

    @classmethod
    def system(cls, content: str) -> Message:
        """Create a system message."""
        return cls(role=MessageRole.SYSTEM, content=content)

    @classmethod
    def tool_response(cls, tool_call_id: str, name: str, content: str, is_error: bool = False) -> Message:
        """Create a tool response message."""
        return cls(
            role=MessageRole.TOOL,
            content=content,
            tool_call_id=tool_call_id,
            name=name,
        )


class EventType(str, Enum):
    """Types of events emitted by the agent."""

    # Session events
    SESSION_START = "session_start"
    SESSION_END = "session_end"

    # Agent loop events
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    TURN_START = "turn_start"
    TURN_END = "turn_end"

    # Message events
    MESSAGE_START = "message_start"
    MESSAGE_DELTA = "message_delta"
    MESSAGE_END = "message_end"

    # Tool events
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_END = "tool_call_end"

    # Thinking events (for extended thinking models)
    THINKING_START = "thinking_start"
    THINKING_DELTA = "thinking_delta"
    THINKING_END = "thinking_end"

    # Error events
    ERROR = "error"


@dataclass
class AgentEvent:
    """An event emitted by the agent."""

    type: EventType
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class AgentConfig(BaseModel):
    """Configuration for the agent."""

    # Model settings
    model: str = "claude-sonnet-4-20250514"
    provider: Literal["anthropic", "openai", "local"] = "anthropic"
    api_key: str | None = None
    base_url: str | None = None

    # Agent behavior
    max_tokens: int = 8192
    temperature: float = 0.0
    system_prompt: str | None = None

    # Tool settings
    max_tool_calls_per_turn: int = 10
    tool_timeout: float = 300.0  # 5 minutes default

    # Session settings
    session_dir: str = ".ds-agent/sessions"
    auto_compact: bool = True
    compact_threshold: int = 100_000  # tokens

    # Extension settings
    extensions_dirs: list[str] = Field(default_factory=lambda: [".ds-agent/extensions"])


@dataclass
class AgentState:
    """Current state of the agent."""

    messages: list[Message] = field(default_factory=list)
    is_running: bool = False
    is_streaming: bool = False
    current_tool_calls: list[ToolCall] = field(default_factory=list)
    token_usage: dict[str, int] = field(default_factory=lambda: {"input": 0, "output": 0})
    error: str | None = None

    # Experiment tracking
    current_experiment: str | None = None
    current_run: str | None = None


# Type aliases for event handlers
EventHandler = Callable[[AgentEvent], None]
AsyncEventHandler = Callable[[AgentEvent], Any]  # Can be async
