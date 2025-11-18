"""Quick test script to verify setup."""

from pathlib import Path

import click


CHECKMARK = "✓"
CROSS = "✗"
WARN = "⚠"


def _header(title: str) -> None:
    click.echo(f"\n{title}")
    click.echo("-" * len(title))


def test_imports() -> bool:
    """Test that all imports work."""
    _header("Checking Python imports")
    try:
        import dspy  # noqa: F401

        click.echo(f"  {CHECKMARK} dspy")
    except ImportError as exc:
        click.echo(f"  {CROSS} dspy: {exc}")
        return False

    try:
        from langgraph.graph import StateGraph  # noqa: F401

        click.echo(f"  {CHECKMARK} langgraph")
    except ImportError as exc:
        click.echo(f"  {CROSS} langgraph: {exc}")
        return False

    try:
        from agent.rag.retrieval import create_retriever  # noqa: F401
        from agent.tools.sqlite_tool import create_sqlite_tool  # noqa: F401
        from agent.graph_hybrid import HybridAgent  # noqa: F401

        click.echo(f"  {CHECKMARK} agent modules")
    except ImportError as exc:
        click.echo(f"  {CROSS} agent modules: {exc}")
        return False

    return True


def test_data() -> bool:
    """Test that data files exist."""
    _header("Looking for local data and docs")

    db_path = Path("data/northwind.sqlite")
    if db_path.exists():
        click.echo(f"  {CHECKMARK} Database: {db_path}")
    else:
        click.echo(f"  {CROSS} Database not found: {db_path}")
        return False

    docs_dir = Path("docs")
    if docs_dir.exists():
        md_files = sorted(docs_dir.glob("*.md"))
        click.echo(f"  {CHECKMARK} Documents: {len(md_files)} files")
        for doc in md_files:
            click.echo(f"    • {doc.name}")
    else:
        click.echo(f"  {CROSS} Docs directory not found: {docs_dir}")
        return False

    eval_file = Path("sample_questions_hybrid_eval.jsonl")
    if eval_file.exists():
        click.echo(f"  {CHECKMARK} Evaluation file: {eval_file}")
    else:
        click.echo(f"  {CROSS} Evaluation file not found: {eval_file}")
        return False

    return True


def test_components() -> bool:
    """Test that components can be instantiated."""
    _header("Smoke-testing core components")

    try:
        from agent.rag.retrieval import create_retriever

        retriever = create_retriever("docs")
        chunks = retriever.retrieve("test query", top_k=1)
        click.echo(f"  {CHECKMARK} RAG Retriever (returned {len(chunks)} chunk(s))")
    except Exception as exc:  # pragma: no cover - diagnostic script
        click.echo(f"  {CROSS} RAG Retriever: {exc}")
        return False

    try:
        from agent.tools.sqlite_tool import create_sqlite_tool

        tool = create_sqlite_tool("data/northwind.sqlite")
        tables = tool.get_table_names()
        click.echo(f"  {CHECKMARK} SQLite Tool (found {len(tables)} tables)")
    except Exception as exc:  # pragma: no cover - diagnostic script
        click.echo(f"  {CROSS} SQLite Tool: {exc}")
        return False

    return True


def test_ollama() -> bool | None:
    """Test Ollama connection."""
    _header("Confirming Ollama is reachable")

    try:
        import requests

        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [model.get("name", "") for model in models]
            if any("phi3.5" in name or "phi" in name.lower() for name in model_names):
                click.echo(
                    f"  {CHECKMARK} Ollama running ({len(model_names)} model(s) loaded)"
                )
                return True
            click.echo(f"  {WARN} Ollama running but phi3.5 model not found")
            click.echo(f"     Available models: {model_names}")
            click.echo("     Run: ollama pull phi3.5:3.8b-mini-instruct-q4_K_M")
            return False
        click.echo(f"  {CROSS} Ollama returned status {response.status_code}")
        return False
    except requests.exceptions.ConnectionError:
        click.echo(f"  {CROSS} Ollama not running or unreachable")
        click.echo("     Start with: ollama serve")
        return False
    except ImportError:
        click.echo(f"  {WARN} Cannot test Ollama (requests not installed)")
        return None
    except Exception as exc:  # pragma: no cover - diagnostic script
        click.echo(f"  {CROSS} Ollama test failed: {exc}")
        return False


def main() -> None:
    click.echo("=" * 60)
    click.echo("Retail Analytics Copilot — Setup Verification")
    click.echo("=" * 60)

    all_ok = True

    if not test_imports():
        all_ok = False

    if not test_data():
        all_ok = False

    if not test_components():
        all_ok = False

    ollama_ok = test_ollama()
    if ollama_ok is False:
        all_ok = False

    click.echo("\n" + "=" * 60)
    if all_ok:
        click.echo(f"{CHECKMARK} All checks passed. You're ready to run the agent!")
        click.echo(
            "\nTry:\n  python run_agent_hybrid.py --batch sample_questions_hybrid_eval.jsonl --out outputs_hybrid.jsonl"
        )
    else:
        raise SystemExit(f"{CROSS} Some checks failed. See notes above.")


if __name__ == "__main__":
    main()

