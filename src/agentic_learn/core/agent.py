"""Main agent class with the agent loop."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

from agentic_learn.core.extension import ExtensionManager
from agentic_learn.core.tool import Tool, ToolContext
from agentic_learn.core.types import (
    AgentConfig,
    AgentEvent,
    AgentState,
    AsyncEventHandler,
    EventType,
    Message,
    MessageRole,
    ToolCall,
    ToolResult,
)


@dataclass
class Agent:
    """The main ML/DS agent.

    Implements a minimal agent loop inspired by pi-mono:
    - Outer loop: continues when follow-up messages arrive
    - Inner loop: processes tool calls between assistant responses
    - No max steps - loops until the agent says it's done
    """

    config: AgentConfig
    state: AgentState = field(default_factory=AgentState)

    # Internal state
    _tools: dict[str, Tool] = field(default_factory=dict)
    _event_handlers: dict[EventType, list[AsyncEventHandler]] = field(default_factory=dict)
    _message_queue: asyncio.Queue[str] = field(default_factory=asyncio.Queue)
    _abort_event: asyncio.Event = field(default_factory=asyncio.Event)
    _extension_manager: ExtensionManager | None = field(default=None)
    _llm_client: Any = field(default=None)

    def __post_init__(self) -> None:
        """Initialize the agent after dataclass creation."""
        self._extension_manager = ExtensionManager(agent=self)

    # Tool management

    def register_tool(self, tool: Tool) -> None:
        """Register a tool with the agent."""
        self._tools[tool.name] = tool

    def unregister_tool(self, name: str) -> None:
        """Remove a tool from the agent."""
        if name in self._tools:
            del self._tools[name]

    def get_tool(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_tools(self) -> list[Tool]:
        """Get all registered tools."""
        return list(self._tools.values())

    # Event handling

    def on(self, event_type: EventType, handler: AsyncEventHandler) -> None:
        """Register an event handler."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)

    async def _emit(self, event_type: EventType, data: dict[str, Any] | None = None) -> None:
        """Emit an event to all handlers."""
        event = AgentEvent(type=event_type, data=data or {})

        # Notify local handlers
        for handler in self._event_handlers.get(event_type, []):
            try:
                result = handler(event)
                if hasattr(result, "__await__"):
                    await result
            except Exception as e:
                print(f"Event handler error: {e}")

        # Notify extensions
        if self._extension_manager:
            await self._extension_manager.emit_event(event)

    # Message handling

    async def queue_message(self, content: str) -> None:
        """Queue a message to be processed."""
        await self._message_queue.put(content)

    def abort(self) -> None:
        """Abort the current operation."""
        self._abort_event.set()

    def _reset_abort(self) -> None:
        """Reset the abort event for a new run."""
        self._abort_event.clear()

    # LLM client

    def _get_llm_client(self) -> Any:
        """Get or create the LLM client."""
        if self._llm_client is not None:
            return self._llm_client

        if self.config.provider == "anthropic":
            try:
                import anthropic

                api_key = self.config.api_key or os.environ.get("ANTHROPIC_API_KEY")
                self._llm_client = anthropic.AsyncAnthropic(api_key=api_key)
            except ImportError:
                raise ImportError("anthropic package required: pip install anthropic")

        elif self.config.provider == "openai":
            try:
                import openai

                api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY")
                self._llm_client = openai.AsyncOpenAI(
                    api_key=api_key,
                    base_url=self.config.base_url,
                )
            except ImportError:
                raise ImportError("openai package required: pip install openai")

        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")

        return self._llm_client

    def _get_tools_schema(self) -> list[dict[str, Any]]:
        """Get tool schemas for the LLM."""
        if self.config.provider == "anthropic":
            return [tool.get_definition().to_anthropic_schema() for tool in self._tools.values()]
        else:
            return [tool.get_definition().to_openai_schema() for tool in self._tools.values()]

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the agent."""
        base_prompt = self.config.system_prompt or self._default_system_prompt()

        # Add tool descriptions
        tool_docs = "\n".join(
            f"- {tool.name}: {tool.description}" for tool in self._tools.values()
        )

        return f"{base_prompt}\n\nAvailable tools:\n{tool_docs}"

    def _default_system_prompt(self) -> str:
        """Default system prompt for ML/DS agent."""
        return """You are an expert ML/DS assistant. You help users with:
- Data exploration and analysis
- Model training and evaluation
- Experiment tracking
- GPU resource management
- Research and paper review

You have access to tools for executing Python code, managing data, tracking experiments,
and monitoring compute resources. Use these tools to help the user accomplish their goals.

When working on ML tasks:
1. First understand what the user wants to achieve
2. Explore the data and environment as needed
3. Execute code in small, testable steps
4. Track important metrics and results
5. Explain your findings and recommendations

Be concise but thorough. Show code and results when relevant."""

    # Core agent loop

    async def run(self, initial_message: str | None = None) -> AsyncIterator[AgentEvent]:
        """Run the agent loop.

        This is the main entry point. The loop continues until:
        - The assistant responds without tool calls
        - And there are no pending messages in the queue

        Yields AgentEvent objects for streaming updates.
        """
        self._reset_abort()
        self.state.is_running = True
        self.state.error = None

        await self._emit(EventType.AGENT_START)
        yield AgentEvent(type=EventType.AGENT_START)

        try:
            # Add initial message if provided
            if initial_message:
                self.state.messages.append(Message.user(initial_message))

            # Main agent loop
            while not self._abort_event.is_set():
                # Check for queued messages
                while not self._message_queue.empty():
                    msg = await self._message_queue.get()
                    self.state.messages.append(Message.user(msg))

                # If no messages, we're done
                if not self.state.messages or self.state.messages[-1].role != MessageRole.USER:
                    # Check if there's a pending message
                    try:
                        msg = await asyncio.wait_for(self._message_queue.get(), timeout=0.1)
                        self.state.messages.append(Message.user(msg))
                    except asyncio.TimeoutError:
                        break

                # Run a turn
                async for event in self._run_turn():
                    yield event

                # If the last message has no tool calls, check for follow-up
                last_msg = self.state.messages[-1] if self.state.messages else None
                if last_msg and last_msg.role == MessageRole.ASSISTANT and not last_msg.tool_calls:
                    # Check for follow-up messages
                    try:
                        msg = await asyncio.wait_for(self._message_queue.get(), timeout=0.1)
                        self.state.messages.append(Message.user(msg))
                    except asyncio.TimeoutError:
                        # No follow-up, we're done
                        break

        except Exception as e:
            self.state.error = str(e)
            await self._emit(EventType.ERROR, {"error": str(e)})
            yield AgentEvent(type=EventType.ERROR, data={"error": str(e)})

        finally:
            self.state.is_running = False
            await self._emit(EventType.AGENT_END)
            yield AgentEvent(type=EventType.AGENT_END)

    async def _run_turn(self) -> AsyncIterator[AgentEvent]:
        """Run a single turn of the agent loop."""
        await self._emit(EventType.TURN_START)
        yield AgentEvent(type=EventType.TURN_START)

        try:
            # Get LLM response
            async for event in self._get_llm_response():
                yield event

            # Process tool calls if any
            last_msg = self.state.messages[-1] if self.state.messages else None
            if last_msg and last_msg.tool_calls:
                async for event in self._execute_tool_calls(last_msg.tool_calls):
                    yield event

                # Continue the turn with tool results
                async for event in self._run_turn():
                    yield event

        finally:
            await self._emit(EventType.TURN_END)
            yield AgentEvent(type=EventType.TURN_END)

    async def _get_llm_response(self) -> AsyncIterator[AgentEvent]:
        """Get a response from the LLM."""
        client = self._get_llm_client()
        self.state.is_streaming = True

        yield AgentEvent(type=EventType.MESSAGE_START)

        try:
            if self.config.provider == "anthropic":
                async for event in self._stream_anthropic(client):
                    yield event
            else:
                async for event in self._stream_openai(client):
                    yield event

        finally:
            self.state.is_streaming = False
            yield AgentEvent(type=EventType.MESSAGE_END)

    async def _stream_anthropic(self, client: Any) -> AsyncIterator[AgentEvent]:
        """Stream response from Anthropic."""
        # Convert messages to Anthropic format
        messages = []
        for msg in self.state.messages:
            if msg.role == MessageRole.USER:
                messages.append({"role": "user", "content": msg.content})
            elif msg.role == MessageRole.ASSISTANT:
                content = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        content.append({
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        })
                messages.append({"role": "assistant", "content": content or msg.content})
            elif msg.role == MessageRole.TOOL:
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": msg.content,
                    }],
                })

        tools = self._get_tools_schema() if self._tools else []

        # Make the API call
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "system": self._get_system_prompt(),
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools

        content_text = ""
        tool_calls: list[ToolCall] = []

        async with client.messages.stream(**kwargs) as stream:
            async for event in stream:
                if hasattr(event, "type"):
                    if event.type == "content_block_delta":
                        if hasattr(event.delta, "text"):
                            content_text += event.delta.text
                            yield AgentEvent(
                                type=EventType.MESSAGE_DELTA,
                                data={"text": event.delta.text},
                            )
                        elif hasattr(event.delta, "partial_json"):
                            yield AgentEvent(
                                type=EventType.MESSAGE_DELTA,
                                data={"tool_input": event.delta.partial_json},
                            )

                    elif event.type == "content_block_start":
                        if hasattr(event.content_block, "type"):
                            if event.content_block.type == "tool_use":
                                tool_calls.append(
                                    ToolCall(
                                        id=event.content_block.id,
                                        name=event.content_block.name,
                                        arguments={},
                                    )
                                )

                    elif event.type == "content_block_stop":
                        pass

            # Get final message
            response = await stream.get_final_message()

            # Update token usage
            self.state.token_usage["input"] += response.usage.input_tokens
            self.state.token_usage["output"] += response.usage.output_tokens

            # Extract tool calls from final response
            for block in response.content:
                if block.type == "tool_use":
                    # Find and update the tool call
                    for tc in tool_calls:
                        if tc.id == block.id:
                            tc.arguments = block.input
                            break
                    else:
                        tool_calls.append(
                            ToolCall(id=block.id, name=block.name, arguments=block.input)
                        )

        # Add assistant message
        self.state.messages.append(
            Message.assistant(content_text, tool_calls if tool_calls else None)
        )

    async def _stream_openai(self, client: Any) -> AsyncIterator[AgentEvent]:
        """Stream response from OpenAI."""
        # Convert messages to OpenAI format
        messages = [{"role": "system", "content": self._get_system_prompt()}]

        for msg in self.state.messages:
            if msg.role == MessageRole.USER:
                messages.append({"role": "user", "content": msg.content})
            elif msg.role == MessageRole.ASSISTANT:
                m: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
                if msg.tool_calls:
                    m["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                        }
                        for tc in msg.tool_calls
                    ]
                messages.append(m)
            elif msg.role == MessageRole.TOOL:
                messages.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content,
                })

        tools = self._get_tools_schema() if self._tools else None

        # Make the API call
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": messages,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools

        content_text = ""
        tool_calls: list[ToolCall] = []
        current_tool_call: dict[str, Any] | None = None

        async for chunk in await client.chat.completions.create(**kwargs):
            delta = chunk.choices[0].delta if chunk.choices else None

            if delta:
                if delta.content:
                    content_text += delta.content
                    yield AgentEvent(
                        type=EventType.MESSAGE_DELTA,
                        data={"text": delta.content},
                    )

                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        if tc_delta.id:
                            # New tool call
                            current_tool_call = {
                                "id": tc_delta.id,
                                "name": tc_delta.function.name if tc_delta.function else "",
                                "arguments": "",
                            }
                            tool_calls.append(current_tool_call)
                        elif current_tool_call and tc_delta.function:
                            if tc_delta.function.arguments:
                                current_tool_call["arguments"] += tc_delta.function.arguments

        # Parse tool call arguments
        parsed_tool_calls = []
        for tc in tool_calls:
            try:
                args = json.loads(tc["arguments"]) if tc["arguments"] else {}
            except json.JSONDecodeError:
                args = {}
            parsed_tool_calls.append(ToolCall(id=tc["id"], name=tc["name"], arguments=args))

        # Add assistant message
        self.state.messages.append(
            Message.assistant(content_text, parsed_tool_calls if parsed_tool_calls else None)
        )

    async def _execute_tool_calls(self, tool_calls: list[ToolCall]) -> AsyncIterator[AgentEvent]:
        """Execute a list of tool calls."""
        for tc in tool_calls:
            if self._abort_event.is_set():
                break

            await self._emit(EventType.TOOL_CALL_START, {"tool": tc.name, "id": tc.id})
            yield AgentEvent(
                type=EventType.TOOL_CALL_START,
                data={"tool": tc.name, "id": tc.id, "arguments": tc.arguments},
            )

            tool = self._tools.get(tc.name)
            if tool is None:
                result = ToolResult(
                    tool_call_id=tc.id,
                    content=f"Error: Unknown tool '{tc.name}'",
                    is_error=True,
                )
            else:
                # Validate arguments
                is_valid, error = tool.validate_args(tc.arguments)
                if not is_valid:
                    result = ToolResult(
                        tool_call_id=tc.id,
                        content=f"Error: {error}",
                        is_error=True,
                    )
                else:
                    # Execute the tool
                    ctx = ToolContext(
                        cwd=os.getcwd(),
                        agent=self,
                        abort_signal=self._abort_event,
                    )
                    try:
                        result = await asyncio.wait_for(
                            tool.execute(ctx, **tc.arguments),
                            timeout=self.config.tool_timeout,
                        )
                        result.tool_call_id = tc.id
                    except asyncio.TimeoutError:
                        result = ToolResult(
                            tool_call_id=tc.id,
                            content=f"Error: Tool '{tc.name}' timed out",
                            is_error=True,
                        )
                    except Exception as e:
                        result = ToolResult(
                            tool_call_id=tc.id,
                            content=f"Error: {str(e)}",
                            is_error=True,
                        )

            # Add tool result message
            content = result.content if isinstance(result.content, str) else json.dumps(result.content)
            self.state.messages.append(
                Message.tool_response(tc.id, tc.name, content, result.is_error)
            )

            await self._emit(EventType.TOOL_CALL_END, {"tool": tc.name, "id": tc.id})
            yield AgentEvent(
                type=EventType.TOOL_CALL_END,
                data={"tool": tc.name, "id": tc.id, "result": content, "is_error": result.is_error},
            )

    # Extension management

    def load_extensions(self, directories: list[str] | None = None) -> None:
        """Load extensions from directories."""
        if self._extension_manager is None:
            self._extension_manager = ExtensionManager(agent=self)

        dirs = directories or self.config.extensions_dirs
        for dir_path in dirs:
            path = Path(dir_path).expanduser()
            if path.exists():
                self._extension_manager.load_from_directory(path)

    def load_extension(self, extension: Any) -> None:
        """Load a single extension."""
        if self._extension_manager is None:
            self._extension_manager = ExtensionManager(agent=self)

        from agentic_learn.core.extension import Extension

        if isinstance(extension, Extension):
            self._extension_manager.load_extension(extension)
        elif isinstance(extension, type) and issubclass(extension, Extension):
            self._extension_manager.load_extension(extension())
