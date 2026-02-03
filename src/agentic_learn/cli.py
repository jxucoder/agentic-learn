"""CLI entry point for the ML/DS coding agent."""

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
from rich.table import Table
from rich.text import Text

from agentic_learn.core.agent import Agent
from agentic_learn.core.types import AgentConfig, EventType
from agentic_learn.tools import get_tools, list_tools, TOOL_INFO

app = typer.Typer(
    name="ds-agent",
    help="ML/DS coding agent with tiered tools",
    no_args_is_help=False,
)
console = Console()


def create_agent(
    model: str = "claude-sonnet-4-20250514",
    provider: str = "anthropic",
    api_key: str | None = None,
    tier: int = 2,
) -> Agent:
    """Create and configure the agent.

    Args:
        model: Model to use
        provider: LLM provider
        api_key: API key
        tier: Tool tier (1=core, 2=+DS essentials, 3=+advanced)
    """
    config = AgentConfig(
        model=model,
        provider=provider,  # type: ignore
        api_key=api_key,
    )

    agent = Agent(config=config)

    # Register tools based on tier
    for tool in get_tools(tier=tier):
        agent.register_tool(tool)

    # Load extensions
    agent.load_extensions()

    return agent


async def run_interactive(agent: Agent) -> None:
    """Run the agent in interactive mode."""
    # Build tool summary by tier
    tools = agent.get_tools()
    tool_names = [t.name for t in tools]

    tier1 = [n for n in tool_names if TOOL_INFO.get(n, {}).get("tier") == 1]
    tier2 = [n for n in tool_names if TOOL_INFO.get(n, {}).get("tier") == 2]
    tier3 = [n for n in tool_names if TOOL_INFO.get(n, {}).get("tier") == 3]

    tool_summary = f"[dim]Core: {', '.join(tier1)}[/dim]"
    if tier2:
        tool_summary += f"\n[dim]DS: {', '.join(tier2)}[/dim]"
    if tier3:
        tool_summary += f"\n[dim]Advanced: {', '.join(tier3)}[/dim]"

    console.print(
        Panel(
            f"[bold blue]DS Coding Agent[/bold blue]\n"
            f"Type your message or /help for commands.\n\n"
            f"{tool_summary}",
            title="Welcome",
            border_style="blue",
        )
    )

    while True:
        try:
            # Get user input
            user_input = Prompt.ask("\n[bold green]You[/bold green]")

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
  /tools    - List available tools by tier
  /clear    - Clear conversation history
  /status   - Show agent status
  /quit     - Exit the agent

[bold]Tool Tiers:[/bold]
  Tier 1 (Core)    - read, write, edit, bash, python
  Tier 2 (DS)      - gpu, data, experiment
  Tier 3 (Advanced) - jobs, tune, viz, notebook, repro""",
                title="Help",
            )
        )
        return True

    elif cmd == "/tools":
        tools = agent.get_tools()
        tool_names = {t.name for t in tools}

        table = Table(title="Available Tools")
        table.add_column("Tier", style="cyan")
        table.add_column("Tool", style="green")
        table.add_column("Description")
        table.add_column("Loaded", style="yellow")

        for name, info in TOOL_INFO.items():
            tier_name = {1: "Core", 2: "DS", 3: "Advanced"}[info["tier"]]
            loaded = "✓" if name in tool_names else "-"
            table.add_row(tier_name, name, info["description"], loaded)

        console.print(table)
        return True

    elif cmd == "/clear":
        agent.state.messages.clear()
        console.print("[dim]Conversation cleared.[/dim]")
        return True

    elif cmd == "/status":
        state = agent.state
        tools = agent.get_tools()
        console.print(
            Panel(
                f"""Messages: {len(state.messages)}
Tools: {len(tools)}
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

    async for event in agent.run(message):
        if event.type == EventType.MESSAGE_DELTA:
            text = event.data.get("text", "")
            if text:
                response_text += text
                console.print(text, end="")

        elif event.type == EventType.TOOL_CALL_START:
            tool_name = event.data.get("tool", "unknown")
            console.print(f"\n[dim]→ {tool_name}[/dim]", end="")

        elif event.type == EventType.TOOL_CALL_END:
            is_error = event.data.get("is_error", False)
            result = event.data.get("result", "")

            if is_error:
                console.print(f" [red]✗[/red]")
            else:
                # Show brief result
                brief = result.split("\n")[0][:60]
                console.print(f" [green]✓[/green] [dim]{brief}[/dim]")

        elif event.type == EventType.MESSAGE_END:
            if response_text:
                console.print()

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
        print()


@app.command()
def main(
    message: Optional[str] = typer.Argument(None, help="Message to send (runs single turn if provided)"),
    model: str = typer.Option("claude-sonnet-4-20250514", "--model", "-m", help="Model to use"),
    provider: str = typer.Option("anthropic", "--provider", "-p", help="LLM provider"),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="API key (or use env var)"),
    tier: int = typer.Option(2, "--tier", "-t", help="Tool tier: 1=core, 2=+DS (default), 3=+advanced"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
    list_tools_flag: bool = typer.Option(False, "--list-tools", "-l", help="List available tools and exit"),
    version: bool = typer.Option(False, "--version", "-v", help="Show version"),
) -> None:
    """DS Coding Agent - Write code and run ML/DS workflows.

    Tool Tiers:
      1 = Core only (read, write, edit, bash, python)
      2 = Core + DS essentials (gpu, data, experiment) [default]
      3 = All tools (+ jobs, tune, viz, notebook, repro)

    Examples:
      ds-agent                           # Interactive mode
      ds-agent "read model.py"           # Single command
      ds-agent -t 1 "write hello.py"     # Core tools only
      ds-agent -t 3 "tune hyperparameters" # All tools
    """
    if version:
        from agentic_learn import __version__
        print(f"ds-agent version {__version__}")
        return

    if list_tools_flag:
        print(list_tools())
        return

    if tier not in (1, 2, 3):
        console.print(f"[red]Invalid tier: {tier}. Use 1, 2, or 3.[/red]")
        raise typer.Exit(1)

    agent = create_agent(model=model, provider=provider, api_key=api_key, tier=tier)

    if message:
        asyncio.run(run_single(agent, message, json_output))
    else:
        asyncio.run(run_interactive(agent))


if __name__ == "__main__":
    app()
