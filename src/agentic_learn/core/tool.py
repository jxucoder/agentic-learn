"""Tool base class and decorator for defining agent tools."""

from __future__ import annotations

import asyncio
import inspect
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, ParamSpec, TypeVar, get_type_hints

from pydantic import BaseModel, Field, create_model

from agentic_learn.core.types import ToolResult


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""

    name: str
    type: type
    description: str
    required: bool = True
    default: Any = None


@dataclass
class ToolDefinition:
    """Complete definition of a tool for LLM consumption."""

    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to OpenAI function calling schema."""
        properties = {}
        required = []

        for param in self.parameters:
            prop: dict[str, Any] = {"description": param.description}

            # Map Python types to JSON schema types
            if param.type in (str, type(None)):
                prop["type"] = "string"
            elif param.type in (int,):
                prop["type"] = "integer"
            elif param.type in (float,):
                prop["type"] = "number"
            elif param.type in (bool,):
                prop["type"] = "boolean"
            elif param.type in (list,):
                prop["type"] = "array"
            elif param.type in (dict,):
                prop["type"] = "object"
            else:
                prop["type"] = "string"

            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def to_anthropic_schema(self) -> dict[str, Any]:
        """Convert to Anthropic tool schema."""
        properties = {}
        required = []

        for param in self.parameters:
            prop: dict[str, Any] = {"description": param.description}

            if param.type in (str, type(None)):
                prop["type"] = "string"
            elif param.type in (int,):
                prop["type"] = "integer"
            elif param.type in (float,):
                prop["type"] = "number"
            elif param.type in (bool,):
                prop["type"] = "boolean"
            elif param.type in (list,):
                prop["type"] = "array"
            elif param.type in (dict,):
                prop["type"] = "object"
            else:
                prop["type"] = "string"

            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


class ToolContext:
    """Context passed to tool execution."""

    def __init__(
        self,
        cwd: str,
        agent: Any,  # Agent instance
        abort_signal: asyncio.Event | None = None,
    ):
        self.cwd = cwd
        self.agent = agent
        self.abort_signal = abort_signal or asyncio.Event()

    def is_aborted(self) -> bool:
        """Check if the tool execution should be aborted."""
        return self.abort_signal.is_set()


class Tool(ABC):
    """Base class for agent tools."""

    name: str
    description: str
    parameters: list[ToolParameter] = []

    def __init__(self):
        if not hasattr(self, "name"):
            self.name = self.__class__.__name__.lower()
        if not hasattr(self, "description"):
            self.description = self.__doc__ or "No description provided"

    @abstractmethod
    async def execute(
        self,
        ctx: ToolContext,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute the tool with the given arguments.

        Args:
            ctx: Tool execution context
            **kwargs: Tool-specific arguments

        Returns:
            ToolResult with the execution result
        """
        ...

    def get_definition(self) -> ToolDefinition:
        """Get the tool definition for LLM consumption."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )

    def validate_args(self, args: dict[str, Any]) -> tuple[bool, str | None]:
        """Validate tool arguments.

        Returns:
            Tuple of (is_valid, error_message)
        """
        for param in self.parameters:
            if param.required and param.name not in args:
                return False, f"Missing required parameter: {param.name}"

            if param.name in args:
                value = args[param.name]
                if not isinstance(value, param.type) and value is not None:
                    # Try type coercion for basic types
                    try:
                        if param.type == str:
                            args[param.name] = str(value)
                        elif param.type == int:
                            args[param.name] = int(value)
                        elif param.type == float:
                            args[param.name] = float(value)
                        elif param.type == bool:
                            args[param.name] = bool(value)
                    except (ValueError, TypeError):
                        return False, f"Invalid type for {param.name}: expected {param.type.__name__}"

        return True, None


# Decorator for creating tools from functions

P = ParamSpec("P")
T = TypeVar("T")


def tool(
    name: str | None = None,
    description: str | None = None,
) -> Callable[[Callable[P, T]], Tool]:
    """Decorator to create a Tool from a function.

    Usage:
        @tool(name="my_tool", description="Does something useful")
        async def my_tool(ctx: ToolContext, arg1: str, arg2: int = 10) -> ToolResult:
            return ToolResult(tool_call_id="", content=f"Result: {arg1}, {arg2}")
    """

    def decorator(func: Callable[P, T]) -> Tool:
        # Extract function metadata
        func_name = name or func.__name__
        func_description = description or func.__doc__ or "No description provided"

        # Extract parameters from function signature
        sig = inspect.signature(func)
        hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}

        parameters: list[ToolParameter] = []
        for param_name, param in sig.parameters.items():
            # Skip 'ctx' parameter
            if param_name == "ctx":
                continue

            param_type = hints.get(param_name, str)
            # Handle Optional types
            if hasattr(param_type, "__origin__"):
                if param_type.__origin__ is type(None):
                    param_type = str
                elif hasattr(param_type, "__args__"):
                    param_type = param_type.__args__[0]

            has_default = param.default is not inspect.Parameter.empty
            parameters.append(
                ToolParameter(
                    name=param_name,
                    type=param_type if isinstance(param_type, type) else str,
                    description=f"Parameter: {param_name}",
                    required=not has_default,
                    default=param.default if has_default else None,
                )
            )

        # Create a Tool subclass dynamically
        class FunctionTool(Tool):
            def __init__(self):
                self.name = func_name
                self.description = func_description
                self.parameters = parameters
                self._func = func

            async def execute(self, ctx: ToolContext, **kwargs: Any) -> ToolResult:
                # Apply defaults
                for param in self.parameters:
                    if param.name not in kwargs and param.default is not None:
                        kwargs[param.name] = param.default

                # Call the function
                if asyncio.iscoroutinefunction(self._func):
                    result = await self._func(ctx, **kwargs)
                else:
                    result = self._func(ctx, **kwargs)

                # Ensure we return a ToolResult
                if isinstance(result, ToolResult):
                    return result
                elif isinstance(result, str):
                    return ToolResult(tool_call_id="", content=result)
                elif isinstance(result, dict):
                    return ToolResult(tool_call_id="", content=json.dumps(result))
                else:
                    return ToolResult(tool_call_id="", content=str(result))

        return FunctionTool()

    return decorator
