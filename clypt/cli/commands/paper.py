"""Paper trading command."""

from clypt.cli.commands.backtest import load_strategy_from_file


def handle_paper(args):
    """Run paper trading command."""
    print(f"\n{'='*70}")
    print("PAPER TRADING MODE")
    print(f"{'='*70}\n")

    # Load strategy
    print(f"Loading strategy from {args.strategy}...")
    StrategyClass = load_strategy_from_file(args.strategy)
    strategy = StrategyClass()
    print(f"Strategy: {strategy.name}\n")

    print("Paper trading not yet implemented.")
    print("This will simulate live trading with fake capital.\n")
