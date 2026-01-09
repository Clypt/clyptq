"""CLI commands.

Lazy imports to avoid loading all dependencies when only one command is needed.
"""

__all__ = ["handle_data", "handle_backtest", "handle_paper", "handle_live"]


def __getattr__(name: str):
    """Lazy import handlers."""
    if name == "handle_data":
        from clyptq.cli.commands.data import handle_data
        return handle_data
    elif name == "handle_backtest":
        from clyptq.cli.commands.backtest import handle_backtest
        return handle_backtest
    elif name == "handle_paper":
        from clyptq.cli.commands.paper import handle_paper
        return handle_paper
    elif name == "handle_live":
        from clyptq.cli.commands.live import handle_live
        return handle_live
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
