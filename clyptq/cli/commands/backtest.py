"""Backtest command."""

import importlib.util
import sys
from datetime import datetime, timedelta
from pathlib import Path

from clyptq import EngineMode
from clyptq.analytics.metrics import print_metrics
from clyptq.data.store import DataStore
from clyptq.engine import Engine
from clyptq.execution import BacktestExecutor
from clyptq.risk import CostModel


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


def load_data_for_strategy(strategy, exchange: str, market_type: str) -> DataStore:
    """Load data from disk for strategy symbols."""
    data_dir = Path(__file__).parent.parent.parent.parent / "data"
    data_path = data_dir / market_type / exchange / "1d"

    if not data_path.exists():
        raise FileNotFoundError(
            f"No data at {data_path}. Run: clypt-engine data download --all"
        )

    store = DataStore()

    # Load all available data
    files = sorted(data_path.glob("*.parquet"))
    print(f"\nLoading {len(files)} symbols from {data_path}")

    import pandas as pd

    for filepath in files:
        symbol = filepath.stem.replace("_", "/")
        df = pd.read_parquet(filepath)
        store.add_ohlcv(symbol, df, frequency="1d", source=exchange)

    print(f"Loaded {len(store)} symbols")
    return store


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

    # Load data
    store = load_data_for_strategy(strategy, args.exchange, args.market)

    # Parse dates
    date_range = store.get_date_range()
    if args.start:
        start_date = datetime.strptime(args.start, "%Y-%m-%d")
    else:
        start_date = date_range.end - timedelta(days=60)

    if args.end:
        end_date = datetime.strptime(args.end, "%Y-%m-%d")
    else:
        end_date = date_range.end

    # Setup engine
    cost_model = CostModel(maker_fee=0.001, taker_fee=0.001, slippage_bps=5.0)
    executor = BacktestExecutor(cost_model)

    engine = Engine(
        strategy=strategy,
        data_store=store,
        mode=EngineMode.BACKTEST,
        executor=executor,
        initial_capital=args.capital,
    )

    print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Capital: ${args.capital:,.0f}\n")

    # Run
    result = engine.run_backtest(start=start_date, end=end_date, verbose=True)

    # Print metrics
    print_metrics(result.metrics)
