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


def main():
    app()


if __name__ == "__main__":
    main()
