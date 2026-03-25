"""
CLI for llm-devproxy.

$ llm-devproxy history
$ llm-devproxy search "keyword"
$ llm-devproxy show my_agent
$ llm-devproxy rewind my_agent --step 3
$ llm-devproxy stats
"""

import typer
from typing import Optional
from ..dev_proxy import DevProxy

app = typer.Typer(
    name="llm-devproxy",
    help="LLM development debug layer - every API call recorded, nothing lost.",
    no_args_is_help=True,
)


def _get_proxy(db: str) -> DevProxy:
    return DevProxy(db_path=db)


@app.command()
def history(
    limit: int = typer.Option(20, help="Number of sessions to show"),
    db: str = typer.Option(".llm_devproxy.db", help="Database path"),
):
    """List recent sessions."""
    proxy = _get_proxy(db)
    sessions = proxy.history(limit)
    if not sessions:
        typer.echo("No sessions found.")


@app.command()
def search(
    keyword: str = typer.Argument(..., help="Keyword to search"),
    limit: int = typer.Option(20, help="Max results"),
    db: str = typer.Option(".llm_devproxy.db", help="Database path"),
):
    """Search through all recorded prompts."""
    proxy = _get_proxy(db)
    results = proxy.search(keyword, limit)
    if not results:
        typer.echo(f"No results for '{keyword}'")


@app.command()
def show(
    session: str = typer.Argument(..., help="Session name or ID"),
    branch: str = typer.Option("main", help="Branch name"),
    db: str = typer.Option(".llm_devproxy.db", help="Database path"),
):
    """Show all steps in a session."""
    proxy = _get_proxy(db)
    proxy.show(session, branch)


@app.command()
def rewind(
    session: str = typer.Argument(..., help="Session name or ID"),
    step: int = typer.Option(..., "--step", "-s", help="Step to rewind to"),
    branch: Optional[str] = typer.Option(None, help="New branch name"),
    db: str = typer.Option(".llm_devproxy.db", help="Database path"),
):
    """
    Rewind to a specific step. Original history is preserved.

    Example:
        llm-devproxy rewind my_agent --step 3
        llm-devproxy rewind my_agent --step 3 --branch new_idea
    """
    proxy = _get_proxy(db)
    proxy.rewind(session, step, branch)


@app.command()
def stats(
    db: str = typer.Option(".llm_devproxy.db", help="Database path"),
):
    """Show cost statistics."""
    proxy = _get_proxy(db)
    proxy.stats()


@app.command()
def compress(
    db: str = typer.Option(".llm_devproxy.db", help="Database path"),
):
    """Compress old response bodies (metadata is preserved)."""
    proxy = _get_proxy(db)
    proxy.compress_old()


@app.command()
def start(
    host: str = typer.Option("127.0.0.1", help="Host to bind"),
    port: int = typer.Option(8080, help="Port to listen on"),
    db: str = typer.Option(".llm_devproxy.db", help="Database path"),
    limit: float = typer.Option(1.0, "--limit", "-l", help="Daily cost limit in USD"),
    reload: bool = typer.Option(False, help="Auto-reload on code changes"),
):
    """
    Start the HTTP proxy server.

    Point your LLM client's base_url here:

        OpenAI:    http://localhost:8080/openai/v1
        Anthropic: http://localhost:8080/anthropic/v1
        Gemini:    http://localhost:8080/gemini/v1
    """
    try:
        from ..proxy.server import run_server
        run_server(host=host, port=port, db_path=db, daily_limit_usd=limit, reload=reload)
    except ImportError as e:
        typer.echo(f"❌ {e}", err=True)
        raise typer.Exit(1)


@app.command()
def tag_cmd(
    request_id: str = typer.Argument(..., help="Request ID to tag"),
    tag_value: str = typer.Argument(..., help="Tag to add"),
    db: str = typer.Option(".llm_devproxy.db", help="Database path"),
):
    """Add a tag to a recorded request."""
    proxy = _get_proxy(db)
    proxy.tag(request_id, tag_value)
    typer.echo(f"✅ Tagged {request_id[:8]}... with '{tag_value}'")


@app.command()
def memo_cmd(
    request_id: str = typer.Argument(..., help="Request ID"),
    memo_text: str = typer.Argument(..., help="Memo text"),
    db: str = typer.Option(".llm_devproxy.db", help="Database path"),
):
    """Add a memo to a recorded request."""
    proxy = _get_proxy(db)
    proxy.memo(request_id, memo_text)
    typer.echo(f"✅ Memo added to {request_id[:8]}...")


@app.command()
def export(
    format: str = typer.Option("csv", "--format", "-f", help="Export format: csv / json"),
    output: str = typer.Option("", "--output", "-o", help="Output file path (default: stdout)"),
    session_id: str = typer.Option("", "--session", "-s", help="Filter by session ID"),
    provider: str = typer.Option("", "--provider", "-p", help="Filter by provider"),
    model: str = typer.Option("", "--model", "-m", help="Filter by model"),
    limit: int = typer.Option(0, "--limit", "-n", help="Max records (0=all)"),
    db: str = typer.Option(".llm_devproxy.db", help="Database path"),
):
    """
    Export recorded requests to CSV or JSON.

    Examples:
        llm-devproxy export -f csv -o requests.csv
        llm-devproxy export -f json --session my_agent
        llm-devproxy export -f csv --provider openai --model o1
    """
    from ..core.export import export_requests

    proxy = _get_proxy(db)
    storage = proxy.engine.storage

    records, _ = storage.list_requests(
        provider=provider,
        model=model,
        session_id=session_id,
        limit=limit if limit > 0 else 10000,
        offset=0,
    )

    result = export_requests(records, format=format)

    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(result)
        typer.echo(f"✅ Exported {len(records)} records to {output}")
    else:
        typer.echo(result)


@app.command()
def web(
    host: str = typer.Option("127.0.0.1", help="Host to bind"),
    port: int = typer.Option(8765, help="Port to listen on"),
    db: str = typer.Option(".llm_devproxy.db", help="Database path"),
):
    """Launch the web dashboard."""
    from ..web.app import run
    run(db_path=db, host=host, port=port)


@app.command()
def pricing(
    action: str = typer.Argument("show", help="show / init / reload / source"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Show pricing for a specific model"),
):
    """
    Manage model pricing data.

    Actions:
        show    - Show current pricing (all models or specific model)
        init    - Create local override file (~/.llm_devproxy/pricing.json)
        reload  - Force reload from remote + local
        source  - Show which pricing source is active

    Examples:
        llm-devproxy pricing show
        llm-devproxy pricing show --model o3
        llm-devproxy pricing init
        llm-devproxy pricing source
    """
    from ..core.pricing import PricingManager, create_local_pricing_template
    from ..core.cost_guard import BUILTIN_PRICING, _get_pricing_manager

    if action == "init":
        create_local_pricing_template()
        return

    pm = _get_pricing_manager()

    if action == "source":
        typer.echo(f"📍 Active pricing source: {pm.get_source()}")
        return

    if action == "reload":
        pm.reload()
        typer.echo(f"✅ Pricing reloaded. Source: {pm.get_source()}")
        return

    # show
    if model:
        p = pm.get(model)
        typer.echo(f"Model: {model}")
        typer.echo(f"  Input:     ${p['input']}/1K tokens")
        typer.echo(f"  Output:    ${p['output']}/1K tokens")
        reasoning = p.get('reasoning')
        typer.echo(f"  Reasoning: {'$' + str(reasoning) + '/1K tokens' if reasoning else 'N/A'}")
    else:
        all_pricing = pm.get_all()
        typer.echo(f"📋 {len(all_pricing)} models loaded (source: {pm.get_source()})\n")
        for name, p in sorted(all_pricing.items()):
            r = p.get('reasoning')
            r_str = f"  reasoning=${r}" if r else ""
            typer.echo(f"  {name:30s}  in=${p['input']:<10}  out=${p['output']:<10}{r_str}")


def main():
    app()


if __name__ == "__main__":
    main()
