"""Core agent loop, types, and base classes."""

from agentic_learn.core.agent import Agent
from agentic_learn.core.types import (
    AgentConfig,
    AgentState,
    Message,
    MessageRole,
    ToolCall,
    ToolResult,
    AgentEvent,
    EventType,
)
from agentic_learn.core.tool import Tool, ToolContext, tool
from agentic_learn.core.extension import Extension, ExtensionAPI, ExtensionContext
from agentic_learn.core.session import SessionManager, SessionMetadata
from agentic_learn.core.sandbox import (
    Sandbox,
    SandboxConfig,
    SandboxResult,
    ProcessSandbox,
    DockerSandbox,
)

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentState",
    "Message",
    "MessageRole",
    "ToolCall",
    "ToolResult",
    "AgentEvent",
    "EventType",
    "Tool",
    "ToolContext",
    "tool",
    "Extension",
    "ExtensionAPI",
    "ExtensionContext",
    "SessionManager",
    "SessionMetadata",
    "Sandbox",
    "SandboxConfig",
    "SandboxResult",
    "ProcessSandbox",
    "DockerSandbox",
]
