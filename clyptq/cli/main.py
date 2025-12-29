"""Main CLI entry point."""

import argparse
import sys


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="clypt-engine",
        description="Clypt Trading Engine CLI",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Data command
    data_parser = subparsers.add_parser("data", help="Download market data")
    data_parser.add_argument("action", choices=["download", "list"], help="Action")
    data_parser.add_argument("--exchange", default="binance", help="Exchange (default: binance)")
    data_parser.add_argument("--timeframe", default="1d", help="Timeframe (default: 1d)")
    data_parser.add_argument("--days", type=int, default=90, help="Days (default: 90)")
    data_parser.add_argument("--limit", type=int, default=60, help="Top N symbols (default: 60)")
    data_parser.add_argument("--symbols", nargs="+", help="Specific symbols")
    data_parser.add_argument("--all", action="store_true", dest="download_all", help="Download ALL pairs")
    data_parser.add_argument("--market", default="spot", choices=["spot", "futures"], help="Market type (default: spot)")

    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run backtest")
    backtest_parser.add_argument("strategy", help="Strategy file path (e.g., strategy.py)")
    backtest_parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    backtest_parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    backtest_parser.add_argument("--capital", type=float, default=10000, help="Initial capital (default: 10000)")
    backtest_parser.add_argument("--exchange", default="binance", help="Exchange (default: binance)")
    backtest_parser.add_argument("--market", default="spot", choices=["spot", "futures"], help="Market type (default: spot)")

    # Paper trading command
    paper_parser = subparsers.add_parser("paper", help="Run paper trading")
    paper_parser.add_argument("strategy", help="Strategy file path")
    paper_parser.add_argument("--capital", type=float, default=10000, help="Initial capital (default: 10000)")
    paper_parser.add_argument("--exchange", default="binance", help="Exchange (default: binance)")
    paper_parser.add_argument("--market", default="spot", choices=["spot", "futures"], help="Market type (default: spot)")

    # Live trading command
    live_parser = subparsers.add_parser("live", help="Run live trading")
    live_parser.add_argument("strategy", help="Strategy file path")
    live_parser.add_argument("--exchange", default="binance", help="Exchange (default: binance)")
    live_parser.add_argument("--market", default="spot", choices=["spot", "futures"], help="Market type (default: spot)")
    live_parser.add_argument("--api-key", help="API key")
    live_parser.add_argument("--api-secret", help="API secret")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Route to appropriate command handler
    if args.command == "data":
        from clyptq.cli.commands.data import handle_data
        handle_data(args)
    elif args.command == "backtest":
        from clyptq.cli.commands.backtest import handle_backtest
        handle_backtest(args)
    elif args.command == "paper":
        from clyptq.cli.commands.paper import handle_paper
        handle_paper(args)
    elif args.command == "live":
        from clyptq.cli.commands.live import handle_live
        handle_live(args)


if __name__ == "__main__":
    main()
