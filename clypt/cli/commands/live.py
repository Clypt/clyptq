"""Live trading command."""

from clypt.cli.commands.backtest import load_strategy_from_file


def handle_live(args):
    """Run live trading command."""
    print(f"\n{'='*70}")
    print("LIVE TRADING MODE")
    print(f"{'='*70}\n")

    # Load strategy
    print(f"Loading strategy from {args.strategy}...")
    StrategyClass = load_strategy_from_file(args.strategy)
    strategy = StrategyClass()
    print(f"Strategy: {strategy.name}\n")

    if not args.api_key or not args.api_secret:
        print("Error: --api-key and --api-secret required for live trading")
        return

    print("Live trading not yet implemented.")
    print("This will trade with real money. Use with caution.\n")
