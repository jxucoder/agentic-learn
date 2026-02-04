#!/usr/bin/env python3
"""Example: Using agentic-learn as a DS coding agent.

This shows how to use the agent programmatically (not via CLI).
"""

import asyncio
from agentic_learn.core import Agent, AgentConfig, EventType
from agentic_learn.tools import get_tools


async def basic_example():
    """Basic usage: Ask the agent to write code."""
    print("=" * 60)
    print("Example 1: Basic Code Generation")
    print("=" * 60)

    # Create agent with tier 1 tools (core coding tools)
    config = AgentConfig(
        model="claude-sonnet-4-20250514",
        provider="anthropic",
    )
    agent = Agent(config=config)

    # Register tools
    for tool in get_tools(tier=1):
        agent.register_tool(tool)

    # Run the agent
    async for event in agent.run("Write a Python function to calculate fibonacci numbers"):
        if event.type == EventType.MESSAGE_DELTA:
            print(event.data.get("text", ""), end="", flush=True)
        elif event.type == EventType.TOOL_CALL_START:
            print(f"\n[Using tool: {event.data.get('tool')}]")
        elif event.type == EventType.TOOL_CALL_END:
            print(f"[Tool result: {event.data.get('result', '')[:100]}...]")

    print("\n")


async def ds_workflow_example():
    """DS workflow: Load data and explore it."""
    print("=" * 60)
    print("Example 2: Data Science Workflow")
    print("=" * 60)

    config = AgentConfig(
        model="claude-sonnet-4-20250514",
        provider="anthropic",
    )
    agent = Agent(config=config)

    # Register tier 2 tools (includes data tool)
    for tool in get_tools(tier=2):
        agent.register_tool(tool)

    # Ask agent to explore data
    prompt = """
    I have a CSV file at data/sales.csv. Please:
    1. Read the file to understand its structure
    2. Write Python code to load it with pandas
    3. Show basic statistics
    """

    async for event in agent.run(prompt):
        if event.type == EventType.MESSAGE_DELTA:
            print(event.data.get("text", ""), end="", flush=True)

    print("\n")


async def sandboxed_example():
    """Run code in a sandbox for safety."""
    print("=" * 60)
    print("Example 3: Sandboxed Execution")
    print("=" * 60)

    from agentic_learn.tools import PythonTool
    from agentic_learn.core.tool import ToolContext

    # Create a sandboxed Python tool
    python_tool = PythonTool(sandbox_by_default=True)
    ctx = ToolContext(cwd="/tmp", agent=None)

    # Execute code safely
    result = await python_tool.execute(
        ctx,
        code="""
import sys
print(f"Python version: {sys.version}")
print(f"Running in sandbox: isolated environment")

# This runs with memory/CPU limits
data = [i**2 for i in range(1000)]
print(f"Computed {len(data)} squares")
""",
        timeout=10.0,
    )

    print(f"Output:\n{result.content}")
    print(f"Sandboxed: {result.metadata.get('sandboxed')}")
    print()


async def session_example():
    """Save and resume sessions."""
    print("=" * 60)
    print("Example 4: Session Persistence")
    print("=" * 60)

    from agentic_learn.core import SessionManager, Message

    # Create session manager
    sm = SessionManager()

    # Simulate a conversation
    from agentic_learn.core import AgentState
    state = AgentState()
    state.messages = [
        Message.user("Help me train a neural network"),
        Message.assistant("I'll help you build a neural network. What framework do you prefer - PyTorch or TensorFlow?"),
        Message.user("PyTorch please"),
    ]
    state.token_usage = {"input": 150, "output": 75}

    # Save the session
    meta = sm.save(state, name="Neural Network Project")
    print(f"Saved session: {meta.id}")
    print(f"Name: {meta.name}")
    print(f"Messages: {meta.message_count}")

    # Later: Load it back
    loaded_state, loaded_meta = sm.load(meta.id)
    print(f"\nLoaded session: {loaded_meta.id}")
    print(f"Messages restored: {len(loaded_state.messages)}")

    # List all sessions
    print("\nAll sessions:")
    for s in sm.list_sessions():
        print(f"  {s.id}: {s.name} ({s.message_count} messages)")

    print()


async def custom_tool_example():
    """Create a custom tool."""
    print("=" * 60)
    print("Example 5: Custom Tool")
    print("=" * 60)

    from agentic_learn.core import tool, ToolContext, ToolResult

    # Define a custom tool using the decorator
    @tool(name="fetch_stock", description="Fetch stock price for a symbol")
    async def fetch_stock(ctx: ToolContext, symbol: str) -> ToolResult:
        # In real implementation, call an API
        prices = {"AAPL": 178.50, "GOOGL": 141.25, "MSFT": 378.90}
        price = prices.get(symbol.upper(), None)

        if price:
            return ToolResult(
                tool_call_id="",
                content=f"{symbol.upper()}: ${price:.2f}",
            )
        else:
            return ToolResult(
                tool_call_id="",
                content=f"Unknown symbol: {symbol}",
                is_error=True,
            )

    # Use the tool
    ctx = ToolContext(cwd="/tmp", agent=None)
    result = await fetch_stock.execute(ctx, symbol="AAPL")
    print(f"Stock tool result: {result.content}")

    # Register with agent
    config = AgentConfig(model="claude-sonnet-4-20250514", provider="anthropic")
    agent = Agent(config=config)
    agent.register_tool(fetch_stock)

    print(f"Agent tools: {[t.name for t in agent.get_tools()]}")
    print()


async def extension_example():
    """Create a custom extension."""
    print("=" * 60)
    print("Example 6: Custom Extension")
    print("=" * 60)

    from agentic_learn.core import Extension, ExtensionAPI, ExtensionContext, EventType

    class MetricsExtension(Extension):
        """Extension that tracks agent metrics."""

        name = "metrics"
        description = "Track agent performance metrics"

        def __init__(self):
            super().__init__()
            self.turn_count = 0
            self.tool_calls = 0

        def setup(self, api: ExtensionAPI) -> None:
            api.on(EventType.TURN_START, self.on_turn)
            api.on(EventType.TOOL_CALL_END, self.on_tool_call)

        async def on_turn(self, ctx: ExtensionContext) -> None:
            self.turn_count += 1
            print(f"[Metrics] Turn {self.turn_count}")

        async def on_tool_call(self, ctx: ExtensionContext) -> None:
            self.tool_calls += 1
            print(f"[Metrics] Tool calls: {self.tool_calls}")

    # Load extension
    config = AgentConfig(model="claude-sonnet-4-20250514", provider="anthropic")
    agent = Agent(config=config)
    agent.load_extension(MetricsExtension())

    print(f"Loaded extensions: {list(agent._extension_manager.extensions.keys())}")
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("agentic-learn Examples")
    print("=" * 60 + "\n")

    # Run examples that don't need API keys
    asyncio.run(sandboxed_example())
    asyncio.run(session_example())
    asyncio.run(custom_tool_example())
    asyncio.run(extension_example())

    print("=" * 60)
    print("Examples requiring API key (not run):")
    print("  - basic_example(): Generate code")
    print("  - ds_workflow_example(): Data exploration")
    print("=" * 60)
    print("\nTo run with API:")
    print("  export ANTHROPIC_API_KEY=your-key")
    print("  python examples/usage.py --with-api")


if __name__ == "__main__":
    import sys
    if "--with-api" in sys.argv:
        asyncio.run(basic_example())
        asyncio.run(ds_workflow_example())
    main()
