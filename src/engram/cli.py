"""CLI — the `engram` command-line tool.

Commands:
    engram chat                    — interactive chat with memory
    engram memories list           — list all memories
    engram memories search "query" — semantic search
    engram memories add "content"  — manually add a memory
    engram memories delete <id>    — delete a memory
    engram stats                   — memory statistics
    engram export <file>           — export memories to JSON
    engram import <file>           — import memories from JSON
    engram conflicts               — review contradictions
    engram config set <key> <val>  — set a config value
    engram config show             — show current config
    engram reset                   — wipe all data
"""

from __future__ import annotations

import sys
from dataclasses import asdict

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown

from engram.config import EngramConfig

console = Console()


def _get_client():
    """Lazy import and create MemoryClient."""
    from engram.client import MemoryClient
    return MemoryClient()


# ======================================================================
# Root group
# ======================================================================

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """🧠 Engram — Persistent memory for local LLMs."""
    if ctx.invoked_subcommand is None:
        # Default to showing help
        click.echo(ctx.get_help())


# ======================================================================
# engram chat
# ======================================================================

@cli.command()
@click.option("--model", "-m", default=None, help="LLM model name (default: from config)")
@click.option("--no-memory", is_flag=True, help="Disable memory retrieval and extraction")
def chat(model, no_memory):
    """Interactive chat with persistent memory."""
    client = _get_client()
    model = model or client.config.default_model

    # Health check
    try:
        from engram.llm import EngramConnectionError
        client.llm_client.check_health()
    except EngramConnectionError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)

    console.print(
        Panel(
            f"[bold cyan]Engram Chat[/bold cyan]\n"
            f"Model: [green]{model}[/green] | "
            f"Memories: [green]{client.store.count()}[/green] | "
            f"Memory: [green]{'ON' if not no_memory else 'OFF'}[/green]\n"
            f"Type [yellow]exit[/yellow] or [yellow]quit[/yellow] to end session.",
            border_style="cyan",
        )
    )

    messages = []

    try:
        while True:
            # Get user input
            try:
                user_input = console.input("[bold green]You:[/bold green] ")
            except (EOFError, KeyboardInterrupt):
                break

            user_input = user_input.strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "/quit", "/exit"):
                break

            # Handle special commands within chat
            if user_input.startswith("/"):
                _handle_chat_command(user_input, client)
                continue

            messages.append({"role": "user", "content": user_input})

            # Stream response
            console.print("[bold blue]Engram:[/bold blue] ", end="")
            full_response = []
            try:
                for chunk in client.chat(
                    messages=messages,
                    model=model,
                    stream=True,
                    extract=not no_memory,
                ):
                    print(chunk, end="", flush=True)
                    full_response.append(chunk)
            except Exception as e:
                console.print(f"\n[bold red]Error:[/bold red] {e}")
                messages.pop()  # Remove failed message
                continue

            print()  # Newline after response
            response_text = "".join(full_response)
            messages.append({"role": "assistant", "content": response_text})

    finally:
        client.close()
        console.print("\n[dim]Session ended.[/dim]")


def _handle_chat_command(command: str, client):
    """Handle slash commands within the chat interface."""
    parts = command.strip().split(maxsplit=1)
    cmd = parts[0].lower()

    if cmd == "/memories":
        memories = client.list(include_archived=False)
        if not memories:
            console.print("[dim]No memories stored yet.[/dim]")
        else:
            for m in memories[:10]:
                pin = "📌 " if m.pinned else ""
                console.print(f"  [dim]•[/dim] {pin}{m.content} [dim][{m.type.value}][/dim]")
            if len(memories) > 10:
                console.print(f"  [dim]... and {len(memories) - 10} more[/dim]")
    elif cmd == "/stats":
        stats = client.stats()
        console.print(f"  Active: {stats['active_memories']} | Archived: {stats['archived_memories']} | Pinned: {stats['pinned_memories']}")
    elif cmd == "/help":
        console.print("[dim]/memories — show stored memories[/dim]")
        console.print("[dim]/stats — show memory statistics[/dim]")
        console.print("[dim]/help — show this help[/dim]")
    else:
        console.print(f"[dim]Unknown command: {cmd}. Try /help[/dim]")


# ======================================================================
# engram memories
# ======================================================================

@cli.group()
def memories():
    """Manage stored memories."""
    pass


@memories.command("list")
@click.option("--archived", is_flag=True, help="Include archived memories")
@click.option("--type", "memory_type", default=None, help="Filter by type")
@click.option("--pinned", is_flag=True, help="Show only pinned memories")
def memories_list(archived, memory_type, pinned):
    """List all stored memories with relevance scores."""
    client = _get_client()

    try:
        from engram.decay import DecayEngine
        from engram.models import MemoryType

        decay = DecayEngine(client.store, client.config)

        if pinned:
            all_memories = client.store.list_memories(pinned_only=True)
            scored = [(m, decay.calculate_relevance(m)) for m in all_memories]
        elif memory_type:
            mt = MemoryType(memory_type)
            all_memories = client.store.list_memories(
                include_archived=archived, memory_type=mt
            )
            scored = [(m, decay.calculate_relevance(m)) for m in all_memories]
        else:
            scored = decay.get_all_relevance_scores()
            if archived:
                archived_mems = [
                    m for m in client.store.list_memories(include_archived=True)
                    if m.archived
                ]
                for m in archived_mems:
                    scored.append((m, decay.calculate_relevance(m)))

        if not scored:
            console.print("[dim]No memories found.[/dim]")
            return

        table = Table(title="Stored Memories", show_lines=True)
        table.add_column("ID", style="dim", max_width=8)
        table.add_column("Content", max_width=50)
        table.add_column("Type", style="cyan")
        table.add_column("Importance", justify="right")
        table.add_column("Relevance", justify="right")
        table.add_column("Flags", justify="center")

        for memory, relevance in scored:
            flags = []
            if memory.pinned:
                flags.append("📌")
            if memory.archived:
                flags.append("📦")
            if memory.conflict_candidate:
                flags.append("⚠️")

            rel_color = "green" if relevance > 0.5 else "yellow" if relevance > 0.15 else "red"

            table.add_row(
                memory.id[:8],
                memory.content[:50] + ("..." if len(memory.content) > 50 else ""),
                memory.type.value,
                f"{memory.importance:.2f}",
                f"[{rel_color}]{relevance:.3f}[/{rel_color}]",
                " ".join(flags) if flags else "",
            )

        console.print(table)

    finally:
        client.close()


@memories.command("search")
@click.argument("query")
@click.option("--top-k", "-k", default=5, help="Number of results")
def memories_search(query, top_k):
    """Semantic search over memories."""
    client = _get_client()

    try:
        results = client.search(query, top_k=top_k)

        if not results:
            console.print("[dim]No matching memories found.[/dim]")
            return

        table = Table(title=f'Search: "{query}"')
        table.add_column("Score", justify="right", style="cyan")
        table.add_column("Content", max_width=60)
        table.add_column("Type", style="dim")

        for memory, score in results:
            table.add_row(
                f"{score:.3f}",
                memory.content,
                memory.type.value,
            )

        console.print(table) 

    finally:
        client.close()


@memories.command("add")
@click.argument("content")
@click.option("--type", "memory_type", default="fact", help="Memory type")
@click.option("--importance", "-i", default=0.5, help="Importance (0-1)")
@click.option("--pin", is_flag=True, help="Pin this memory")
def memories_add(content, memory_type, importance, pin):
    """Manually add a memory."""
    client = _get_client()

    try:
        memory = client.add(content, type=memory_type, importance=importance, pin=pin)
        console.print(
            f"[green]✓[/green] Memory stored: [cyan]{memory.id[:8]}[/cyan] "
            f"({memory.type.value}, importance={memory.importance:.2f}"
            f"{', 📌 pinned' if memory.pinned else ''})"
        )
    finally:
        client.close()


@memories.command("delete")
@click.argument("memory_id")
@click.confirmation_option(prompt="Are you sure you want to delete this memory?")
def memories_delete(memory_id):
    """Delete a memory by ID (prefix match)."""
    client = _get_client()

    try:
        # Support prefix matching (user provides first 8 chars)
        all_memories = client.list(include_archived=True)
        matches = [m for m in all_memories if m.id.startswith(memory_id)]

        if not matches:
            console.print(f"[red]No memory found matching ID: {memory_id}[/red]")
            return
        if len(matches) > 1:
            console.print(f"[yellow]Multiple matches — provide more characters:[/yellow]")
            for m in matches:
                console.print(f"  {m.id[:12]}  {m.content[:40]}")
            return

        client.forget(matches[0].id)
        console.print(f"[green]✓[/green] Memory {matches[0].id[:8]} deleted.")
    finally:
        client.close()


@memories.command("restore")
@click.argument("memory_id")
def memories_restore(memory_id):
    """Restore an archived memory."""
    client = _get_client()

    try:
        all_memories = client.list(include_archived=True)
        matches = [m for m in all_memories if m.id.startswith(memory_id) and m.archived]

        if not matches:
            console.print(f"[red]No archived memory found matching ID: {memory_id}[/red]")
            return

        memory = matches[0]
        embedding = client.embedding_engine.embed(memory.content)
        success = client.store.restore(memory.id, embedding)

        if success:
            console.print(f"[green]✓[/green] Memory {memory.id[:8]} restored.")
        else:
            console.print(f"[red]Failed to restore memory.[/red]")
    finally:
        client.close()


# ======================================================================
# engram stats
# ======================================================================

@cli.command()
def stats():
    """Show memory statistics."""
    client = _get_client()

    try:
        s = client.stats()

        table = Table(title="Engram Statistics", show_header=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        table.add_row("Active memories", str(s["active_memories"]))
        table.add_row("Archived memories", str(s["archived_memories"]))
        table.add_row("Total memories", str(s["total_memories"]))
        table.add_row("Pinned memories", str(s["pinned_memories"]))
        table.add_row("Conflict candidates", str(s["conflict_candidates"]))
        table.add_row("Avg importance", f"{s['avg_importance']:.3f}")

        if s["type_breakdown"]:
            table.add_row("", "")
            for mtype, count in s["type_breakdown"].items():
                table.add_row(f"  {mtype}", str(count))

        console.print(table)
    finally:
        client.close()


# ======================================================================
# engram export / import
# ======================================================================

@cli.command("export")
@click.argument("file", type=click.Path())
def export_cmd(file):
    """Export all memories to a JSON file."""
    client = _get_client()
    try:
        count = client.export(file)
        console.print(f"[green]✓[/green] Exported {count} memories to {file}")
    finally:
        client.close()


@cli.command("import")
@click.argument("file", type=click.Path(exists=True))
def import_cmd(file):
    """Import memories from a JSON file."""
    client = _get_client()
    try:
        count = client.import_memories(file)
        console.print(f"[green]✓[/green] Imported {count} memories from {file}")
    finally:
        client.close()


# ======================================================================
# engram conflicts
# ======================================================================

@cli.command()
def conflicts():
    """Review detected contradictions."""
    client = _get_client()

    try:
        records = client.store.list_conflicts()

        if not records:
            console.print("[dim]No conflicts recorded.[/dim]")
            return

        table = Table(title="Conflict Log")
        table.add_column("Time", style="dim", max_width=19)
        table.add_column("Old Memory", max_width=30)
        table.add_column("New Memory", max_width=30)
        table.add_column("Verdict", style="cyan")
        table.add_column("Resolution")

        for r in records[-20:]:  # Show last 20
            table.add_row(
                r.get("timestamp", "")[:19],
                r.get("old_memory_content", "")[:30],
                r.get("new_memory_content", "")[:30],
                r.get("verdict", ""),
                r.get("resolution", ""),
            )

        console.print(table)
    finally:
        client.close()


# ======================================================================
# engram config
# ======================================================================

@cli.group()
def config():
    """View and modify Engram configuration."""
    pass


@config.command("show")
def config_show():
    """Show current configuration."""
    cfg = EngramConfig.load()
    
    table = Table(title="Engram Configuration", show_header=False)
    table.add_column("Key", style="cyan")
    table.add_column("Value")

    for key, value in asdict(cfg).items():
        table.add_row(key, str(value))

    console.print(table)


@config.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key, value):
    """Set a configuration value."""
    cfg = EngramConfig.load()
    try:
        cfg.set_value(key, value)
        console.print(f"[green]✓[/green] Set {key} = {value}")
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")


# ======================================================================
# engram reset
# ======================================================================

@cli.command()
@click.confirmation_option(prompt="This will DELETE all Engram data. Are you sure?")
def reset():
    """Wipe all Engram data (memories, config, logs)."""
    import shutil
    cfg = EngramConfig.load()
    try:
        shutil.rmtree(cfg.data_dir)
        console.print(f"[green]✓[/green] All Engram data deleted from {cfg.data_dir}")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


# ======================================================================
# Entry point
# ======================================================================

if __name__ == "__main__":
    cli()
