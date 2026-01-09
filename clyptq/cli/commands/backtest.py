"""Backtest command."""

import importlib.util
import sys
from datetime import datetime, timedelta
from pathlib import Path

from clyptq.analytics.performance.metrics import print_metrics
from clyptq.data.sources import ParquetSource
from clyptq.trading.engine import BacktestEngine
from clyptq.trading.execution import BacktestExecutor
from clyptq.core.types import CostModel


def load_strategy_from_file(filepath: str):
    """Dynamically load strategy from Python file."""
    spec = importlib.util.spec_from_file_location("strategy_module", filepath)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load strategy from {filepath}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["strategy_module"] = module
    spec.loader.exec_module(module)

    # Find Strategy class in module (exclude imported base classes)
    for name in dir(module):
        obj = getattr(module, name)
        if (
            isinstance(obj, type)
            and hasattr(obj, "__bases__")
            and any("Strategy" in base.__name__ for base in obj.__bases__)
            and obj.__module__ == "strategy_module"  # Only user-defined classes
        ):
            return obj

    raise ValueError(f"No Strategy class found in {filepath}")


def load_data_source(exchange: str, market_type: str) -> ParquetSource:
    """Load data source from disk."""
    # clyptq/cli/commands/backtest.py -> project root
    data_dir = Path(__file__).parent.parent.parent.parent / "data"
    data_path = data_dir / market_type / exchange / "1d"

    if not data_path.exists():
        raise FileNotFoundError(
            f"No data at {data_path}. Run: clypt-engine data download --all"
        )

    source = ParquetSource(path=data_path, timeframe="1d")
    symbols = source.available_symbols()
    print(f"\nLoaded {len(symbols)} symbols from {data_path}")

    return source


def handle_backtest(args):
    """Run backtest command."""
    print(f"\n{'='*70}")
    print("BACKTEST MODE")
    print(f"{'='*70}\n")

    # Load strategy
    print(f"Loading strategy from {args.strategy}...")
    StrategyClass = load_strategy_from_file(args.strategy)
    strategy = StrategyClass()
    print(f"Strategy: {strategy.name}\n")

    # Load data source
    source = load_data_source(args.exchange, args.market)
    symbols = source.available_symbols()

    # Parse dates
    if args.start:
        start_date = datetime.strptime(args.start, "%Y-%m-%d")
    else:
        start_date = datetime.now() - timedelta(days=60)

    if args.end:
        end_date = datetime.strptime(args.end, "%Y-%m-%d")
    else:
        end_date = datetime.now()

    # Setup engine
    cost_model = CostModel(maker_fee=0.001, taker_fee=0.001, slippage_bps=5.0)
    executor = BacktestExecutor(cost_model)

    engine = BacktestEngine(
        strategy=strategy,
        source=source,
        executor=executor,
        initial_capital=args.capital,
    )

    print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Capital: ${args.capital:,.0f}\n")

    # Run backtest
    result = engine.run(
        symbols=symbols,
        start=start_date,
        end=end_date,
        verbose=True,
    )

    # Print metrics
    print_metrics(result.metrics)
