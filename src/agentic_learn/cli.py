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
from agentic_learn.core.session import SessionManager, SessionMetadata
from agentic_learn.core.types import AgentConfig, EventType
from agentic_learn.tools import get_tools, list_tools, TOOL_INFO

# Global session manager
_session_manager: SessionManager | None = None
_current_session_id: str | None = None
_autosave: bool = False


def get_session_manager() -> SessionManager:
    """Get or create the session manager."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager

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


async def run_interactive(agent: Agent, autosave: bool = False) -> None:
    """Run the agent in interactive mode."""
    global _autosave, _current_session_id
    _autosave = autosave

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

    session_info = ""
    if _current_session_id:
        session_info = f"\n[dim]Session: {_current_session_id}[/dim]"
    if autosave:
        session_info += " [dim](autosave)[/dim]"

    console.print(
        Panel(
            f"[bold blue]DS Coding Agent[/bold blue]\n"
            f"Type your message or /help for commands.{session_info}\n\n"
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
                if user_input.lower() in ("/quit", "/exit"):
                    break

            # Run the agent
            console.print()
            await run_agent_turn(agent, user_input)

            # Auto-save if enabled
            if _autosave and agent.state.messages:
                sm = get_session_manager()
                meta = sm.save(agent.state, session_id=_current_session_id)
                _current_session_id = meta.id

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Type /quit to exit.[/yellow]")
        except EOFError:
            break

    # Final save prompt if not autosave
    if not _autosave and agent.state.messages:
        save = Prompt.ask("[dim]Save session before exit?[/dim]", choices=["y", "n"], default="n")
        if save == "y":
            sm = get_session_manager()
            meta = sm.save(agent.state, session_id=_current_session_id)
            console.print(f"[dim]Session saved: {meta.id}[/dim]")

    console.print("[dim]Goodbye![/dim]")


async def handle_command(command: str, agent: Agent) -> bool:
    """Handle a slash command. Returns True if handled."""
    global _current_session_id

    parts = command.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else None

    if cmd == "/help":
        console.print(
            Panel(
                """[bold]Commands:[/bold]
  /help           - Show this help
  /tools          - List available tools by tier
  /clear          - Clear conversation history
  /status         - Show agent status

[bold]Session Commands:[/bold]
  /save [name]    - Save current session
  /load <id>      - Load a session
  /sessions       - List recent sessions
  /delete <id>    - Delete a session

[bold]Exit:[/bold]
  /quit           - Exit the agent

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
        _current_session_id = None
        console.print("[dim]Conversation cleared.[/dim]")
        return True

    elif cmd == "/status":
        state = agent.state
        tools = agent.get_tools()
        session_info = f"Session: {_current_session_id or 'None'}"
        console.print(
            Panel(
                f"""{session_info}
Messages: {len(state.messages)}
Tools: {len(tools)}
Tokens: {state.token_usage}
Error: {state.error or 'None'}""",
                title="Agent Status",
            )
        )
        return True

    elif cmd == "/save":
        sm = get_session_manager()
        meta = sm.save(agent.state, name=arg, session_id=_current_session_id)
        _current_session_id = meta.id
        console.print(f"[green]Session saved:[/green] {meta.id} - {meta.name}")
        return True

    elif cmd == "/load":
        if not arg:
            console.print("[red]Usage: /load <session_id>[/red]")
            return True

        sm = get_session_manager()
        result = sm.load(arg)
        if result is None:
            console.print(f"[red]Session not found: {arg}[/red]")
            return True

        state, meta = result
        agent.state.messages = state.messages
        agent.state.token_usage = state.token_usage
        _current_session_id = meta.id
        console.print(f"[green]Loaded session:[/green] {meta.id} - {meta.name}")
        console.print(f"[dim]{len(state.messages)} messages, tokens: {state.token_usage}[/dim]")
        return True

    elif cmd == "/sessions":
        sm = get_session_manager()
        sessions = sm.list_sessions(limit=10)

        if not sessions:
            console.print("[dim]No saved sessions.[/dim]")
            return True

        table = Table(title="Recent Sessions")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Messages", justify="right")
        table.add_column("Updated", style="dim")

        for s in sessions:
            # Format date
            updated = s.updated_at[:16].replace("T", " ")
            name = s.name[:40] + "..." if len(s.name) > 40 else s.name
            table.add_row(s.id, name, str(s.message_count), updated)

        console.print(table)
        return True

    elif cmd == "/delete":
        if not arg:
            console.print("[red]Usage: /delete <session_id>[/red]")
            return True

        sm = get_session_manager()
        if sm.delete(arg):
            console.print(f"[green]Deleted session: {arg}[/green]")
            if _current_session_id == arg:
                _current_session_id = None
        else:
            console.print(f"[red]Session not found: {arg}[/red]")
        return True

    elif cmd in ("/quit", "/exit"):
        return False  # Signal to exit

    else:
        console.print(f"[red]Unknown command: {cmd}[/red]")
        return True

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
    resume: Optional[str] = typer.Option(None, "--resume", "-r", help="Resume a saved session by ID"),
    autosave: bool = typer.Option(False, "--autosave", "-a", help="Auto-save session after each turn"),
    list_sessions: bool = typer.Option(False, "--sessions", "-s", help="List saved sessions and exit"),
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
      ds-agent --resume abc123           # Resume saved session
      ds-agent --autosave                # Auto-save after each turn
      ds-agent --sessions                # List saved sessions
    """
    global _current_session_id

    if version:
        from agentic_learn import __version__
        print(f"ds-agent version {__version__}")
        return

    if list_tools_flag:
        print(list_tools())
        return

    if list_sessions:
        sm = get_session_manager()
        sessions = sm.list_sessions(limit=20)
        if not sessions:
            print("No saved sessions.")
            return
        print(f"{'ID':<10} {'Messages':>8}  {'Updated':<16}  Name")
        print("-" * 70)
        for s in sessions:
            updated = s.updated_at[:16].replace("T", " ")
            name = s.name[:40] + "..." if len(s.name) > 40 else s.name
            print(f"{s.id:<10} {s.message_count:>8}  {updated:<16}  {name}")
        return

    if tier not in (1, 2, 3):
        console.print(f"[red]Invalid tier: {tier}. Use 1, 2, or 3.[/red]")
        raise typer.Exit(1)

    agent = create_agent(model=model, provider=provider, api_key=api_key, tier=tier)

    # Resume session if specified
    if resume:
        sm = get_session_manager()
        result = sm.load(resume)
        if result is None:
            console.print(f"[red]Session not found: {resume}[/red]")
            raise typer.Exit(1)
        state, meta = result
        agent.state.messages = state.messages
        agent.state.token_usage = state.token_usage
        _current_session_id = meta.id
        console.print(f"[green]Resumed session:[/green] {meta.id} - {meta.name}")

    if message:
        asyncio.run(run_single(agent, message, json_output))
    else:
        asyncio.run(run_interactive(agent, autosave=autosave))


if __name__ == "__main__":
    app()
