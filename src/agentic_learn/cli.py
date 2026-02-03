"""CLI entry point for the ML/DS agent."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.status import Status
from rich.text import Text

from agentic_learn.core.agent import Agent
from agentic_learn.core.types import AgentConfig, EventType
from agentic_learn.tools import get_default_tools

app = typer.Typer(
    name="ds-agent",
    help="Minimal, extensible ML/DS agent",
    no_args_is_help=False,
)
console = Console()


def create_agent(
    model: str = "claude-sonnet-4-20250514",
    provider: str = "anthropic",
    api_key: str | None = None,
) -> Agent:
    """Create and configure the agent."""
    config = AgentConfig(
        model=model,
        provider=provider,  # type: ignore
        api_key=api_key,
    )

    agent = Agent(config=config)

    # Register default tools
    for tool in get_default_tools():
        agent.register_tool(tool)

    # Load extensions
    agent.load_extensions()

    return agent


async def run_interactive(agent: Agent) -> None:
    """Run the agent in interactive mode."""
    console.print(
        Panel(
            "[bold blue]DS Agent[/bold blue] - ML/DS Assistant\n"
            "Type your message or /help for commands. Ctrl+C to exit.",
            title="Welcome",
            border_style="blue",
        )
    )

    # Show available tools
    tools = agent.get_tools()
    tool_names = [t.name for t in tools]
    console.print(f"[dim]Available tools: {', '.join(tool_names)}[/dim]\n")

    while True:
        try:
            # Get user input
            user_input = Prompt.ask("[bold green]You[/bold green]")

            if not user_input.strip():
                continue

            # Handle commands
            if user_input.startswith("/"):
                if await handle_command(user_input, agent):
                    continue
                if user_input == "/quit" or user_input == "/exit":
                    break

            # Run the agent
            console.print()
            await run_agent_turn(agent, user_input)
            console.print()

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Type /quit to exit.[/yellow]")
        except EOFError:
            break

    console.print("[dim]Goodbye![/dim]")


async def handle_command(command: str, agent: Agent) -> bool:
    """Handle a slash command. Returns True if handled."""
    cmd = command.lower().strip()

    if cmd == "/help":
        console.print(
            Panel(
                """[bold]Commands:[/bold]
  /help     - Show this help
  /tools    - List available tools
  /clear    - Clear conversation history
  /status   - Show agent status
  /quit     - Exit the agent

[bold]Tips:[/bold]
  - Ask about your data, models, or experiments
  - Use tools to execute Python code, check GPU status, etc.
  - The agent maintains context across messages""",
                title="Help",
            )
        )
        return True

    elif cmd == "/tools":
        tools = agent.get_tools()
        console.print("\n[bold]Available Tools:[/bold]")
        for tool in tools:
            console.print(f"  [cyan]{tool.name}[/cyan]: {tool.description.split(chr(10))[0]}")
        console.print()
        return True

    elif cmd == "/clear":
        agent.state.messages.clear()
        console.print("[dim]Conversation cleared.[/dim]")
        return True

    elif cmd == "/status":
        state = agent.state
        console.print(
            Panel(
                f"""Messages: {len(state.messages)}
Running: {state.is_running}
Tokens: {state.token_usage}
Error: {state.error or 'None'}""",
                title="Agent Status",
            )
        )
        return True

    elif cmd in ("/quit", "/exit"):
        return False  # Signal to exit

    return False


async def run_agent_turn(agent: Agent, message: str) -> None:
    """Run a single turn with the agent."""
    response_text = ""
    current_tool = None

    async for event in agent.run(message):
        if event.type == EventType.MESSAGE_DELTA:
            text = event.data.get("text", "")
            if text:
                response_text += text
                console.print(text, end="")

        elif event.type == EventType.TOOL_CALL_START:
            tool_name = event.data.get("tool", "unknown")
            current_tool = tool_name
            console.print(f"\n[dim]→ Using tool: {tool_name}[/dim]")

        elif event.type == EventType.TOOL_CALL_END:
            tool_name = event.data.get("tool", "unknown")
            is_error = event.data.get("is_error", False)
            result = event.data.get("result", "")

            if is_error:
                console.print(f"[red]  ✗ Error: {result[:200]}[/red]")
            else:
                # Truncate long results
                if len(result) > 500:
                    result = result[:500] + "..."
                console.print(f"[dim]  ✓ {result[:100]}...[/dim]" if len(result) > 100 else f"[dim]  ✓ {result}[/dim]")

            current_tool = None

        elif event.type == EventType.MESSAGE_END:
            if response_text:
                console.print()  # Newline after response

        elif event.type == EventType.ERROR:
            error = event.data.get("error", "Unknown error")
            console.print(f"\n[red]Error: {error}[/red]")


async def run_single(agent: Agent, message: str, output_json: bool = False) -> None:
    """Run a single message and exit."""
    response_text = ""
    tool_results = []

    async for event in agent.run(message):
        if event.type == EventType.MESSAGE_DELTA:
            text = event.data.get("text", "")
            if text:
                response_text += text
                if not output_json:
                    print(text, end="", flush=True)

        elif event.type == EventType.TOOL_CALL_END:
            tool_results.append({
                "tool": event.data.get("tool"),
                "result": event.data.get("result"),
                "is_error": event.data.get("is_error"),
            })

    if output_json:
        import json
        print(json.dumps({
            "response": response_text,
            "tool_results": tool_results,
        }))
    else:
        print()  # Final newline


@app.command()
def main(
    message: Optional[str] = typer.Argument(None, help="Message to send (runs single turn if provided)"),
    model: str = typer.Option("claude-sonnet-4-20250514", "--model", "-m", help="Model to use"),
    provider: str = typer.Option("anthropic", "--provider", "-p", help="LLM provider"),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="API key (or use env var)"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
    version: bool = typer.Option(False, "--version", "-v", help="Show version"),
) -> None:
    """ML/DS Agent - An extensible assistant for machine learning and data science."""
    if version:
        from agentic_learn import __version__
        print(f"ds-agent version {__version__}")
        return

    agent = create_agent(model=model, provider=provider, api_key=api_key)

    if message:
        # Single turn mode
        asyncio.run(run_single(agent, message, json_output))
    else:
        # Interactive mode
        asyncio.run(run_interactive(agent))


if __name__ == "__main__":
    app()
