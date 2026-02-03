"""Extension system for adding capabilities to the agent."""

from __future__ import annotations

import importlib.util
import sys
from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from agentic_learn.core.tool import Tool
from agentic_learn.core.types import AgentEvent, EventType, Message

if TYPE_CHECKING:
    from agentic_learn.core.agent import Agent


@dataclass
class ExtensionContext:
    """Context available to extensions during event handling."""

    agent: Agent
    cwd: str
    event: AgentEvent | None = None

    # Session info
    session_id: str | None = None
    experiment_id: str | None = None
    run_id: str | None = None

    def is_idle(self) -> bool:
        """Check if agent is idle."""
        return not self.agent.state.is_running

    def abort(self) -> None:
        """Abort current agent operation."""
        self.agent.abort()

    def get_messages(self) -> list[Message]:
        """Get current conversation messages."""
        return self.agent.state.messages.copy()

    def get_token_usage(self) -> dict[str, int]:
        """Get current token usage."""
        return self.agent.state.token_usage.copy()

    async def send_message(self, content: str) -> None:
        """Queue a message to be sent to the agent."""
        await self.agent.queue_message(content)


EventHandler = Callable[[ExtensionContext], None]
AsyncEventHandler = Callable[[ExtensionContext], Any]


@dataclass
class Command:
    """A command that can be invoked by the user."""

    name: str
    description: str
    handler: Callable[[ExtensionContext, list[str]], Any]


class ExtensionAPI:
    """API exposed to extensions for registering capabilities."""

    def __init__(self, extension: Extension, agent: Agent):
        self._extension = extension
        self._agent = agent
        self._event_handlers: dict[EventType, list[AsyncEventHandler]] = {}
        self._tools: list[Tool] = []
        self._commands: dict[str, Command] = {}

    def on(self, event_type: EventType, handler: AsyncEventHandler) -> None:
        """Register an event handler.

        Available events:
        - SESSION_START: Fired when session loads
        - SESSION_END: Fired when session ends
        - AGENT_START: Before agent loop starts
        - AGENT_END: After agent loop ends
        - TURN_START: Before each LLM turn
        - TURN_END: After each LLM turn
        - TOOL_CALL_START: Before tool execution
        - TOOL_CALL_END: After tool execution
        - ERROR: On any error
        """
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)

    def register_tool(self, tool: Tool) -> None:
        """Register a custom tool."""
        self._tools.append(tool)

    def register_command(
        self,
        name: str,
        description: str,
        handler: Callable[[ExtensionContext, list[str]], Any],
    ) -> None:
        """Register a slash command (e.g., /mycommand)."""
        self._commands[name] = Command(name=name, description=description, handler=handler)

    def get_tools(self) -> list[Tool]:
        """Get all registered tools."""
        return self._tools.copy()

    def get_commands(self) -> dict[str, Command]:
        """Get all registered commands."""
        return self._commands.copy()

    def get_event_handlers(self, event_type: EventType) -> list[AsyncEventHandler]:
        """Get handlers for an event type."""
        return self._event_handlers.get(event_type, [])


class Extension(ABC):
    """Base class for extensions.

    Extensions can:
    - Register custom tools
    - Hook into agent events
    - Add slash commands
    - Modify system prompts
    - Track experiments
    """

    name: str = "unnamed"
    description: str = "No description"
    version: str = "0.1.0"

    def __init__(self):
        if not hasattr(self, "name") or self.name == "unnamed":
            self.name = self.__class__.__name__.lower().replace("extension", "")

    def setup(self, api: ExtensionAPI) -> None:
        """Set up the extension. Override this to register tools/events/commands.

        Args:
            api: The extension API for registering capabilities
        """
        pass

    def teardown(self) -> None:
        """Clean up when extension is unloaded. Override if needed."""
        pass


@dataclass
class ExtensionManager:
    """Manages loading and lifecycle of extensions."""

    agent: Agent
    extensions: dict[str, Extension] = field(default_factory=dict)
    apis: dict[str, ExtensionAPI] = field(default_factory=dict)

    def load_extension(self, extension: Extension) -> None:
        """Load and initialize an extension."""
        api = ExtensionAPI(extension, self.agent)
        extension.setup(api)

        self.extensions[extension.name] = extension
        self.apis[extension.name] = api

        # Register tools with agent
        for tool in api.get_tools():
            self.agent.register_tool(tool)

    def load_from_path(self, path: Path) -> None:
        """Load an extension from a Python file."""
        if not path.exists():
            raise FileNotFoundError(f"Extension not found: {path}")

        # Load the module
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load extension: {path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[path.stem] = module
        spec.loader.exec_module(module)

        # Look for Extension subclass or setup function
        if hasattr(module, "extension"):
            ext = module.extension
            if isinstance(ext, Extension):
                self.load_extension(ext)
            elif callable(ext):
                # It's a factory function
                self.load_extension(ext())
        elif hasattr(module, "setup"):
            # Functional extension style
            class FunctionalExtension(Extension):
                name = path.stem
                description = getattr(module, "__doc__", "") or f"Extension from {path.name}"

                def setup(self, api: ExtensionAPI) -> None:
                    module.setup(api)

            self.load_extension(FunctionalExtension())
        else:
            # Look for Extension subclasses
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, Extension)
                    and attr is not Extension
                ):
                    self.load_extension(attr())
                    break

    def load_from_directory(self, directory: Path) -> None:
        """Load all extensions from a directory."""
        if not directory.exists():
            return

        # Load single-file extensions
        for path in directory.glob("*.py"):
            if path.name.startswith("_"):
                continue
            try:
                self.load_from_path(path)
            except Exception as e:
                print(f"Failed to load extension {path}: {e}")

        # Load directory extensions (with index.py)
        for path in directory.iterdir():
            if path.is_dir() and (path / "index.py").exists():
                try:
                    self.load_from_path(path / "index.py")
                except Exception as e:
                    print(f"Failed to load extension {path}: {e}")

    def unload_extension(self, name: str) -> None:
        """Unload an extension."""
        if name in self.extensions:
            self.extensions[name].teardown()
            del self.extensions[name]
            del self.apis[name]

    def get_all_tools(self) -> list[Tool]:
        """Get all tools from all extensions."""
        tools: list[Tool] = []
        for api in self.apis.values():
            tools.extend(api.get_tools())
        return tools

    def get_all_commands(self) -> dict[str, Command]:
        """Get all commands from all extensions."""
        commands: dict[str, Command] = {}
        for api in self.apis.values():
            commands.update(api.get_commands())
        return commands

    async def emit_event(self, event: AgentEvent) -> None:
        """Emit an event to all extensions."""
        ctx = ExtensionContext(
            agent=self.agent,
            cwd=self.agent.config.session_dir,
            event=event,
        )

        for api in self.apis.values():
            for handler in api.get_event_handlers(event.type):
                try:
                    result = handler(ctx)
                    if hasattr(result, "__await__"):
                        await result
                except Exception as e:
                    print(f"Extension event handler error: {e}")
