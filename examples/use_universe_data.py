"""
Example: Using Pre-Downloaded Universe Data

Shows how to use the pre-downloaded market data instead of
downloading on every run. This is much faster for development
and testing with large universes (up to 50 symbols).

Pre-download data using:
    python -m clypt.cli.data download --limit 60

List downloaded data:
    python -m clypt.cli.data list
"""

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from clypt import Constraints, CostModel, EngineMode
from clypt.analytics.metrics import print_metrics
from clypt.data.store import DataStore
from clypt.engine import Engine
from clypt.execution import BacktestExecutor
from clypt.factors.library.momentum import MomentumFactor
from clypt.factors.library.volatility import VolatilityFactor
from clypt.portfolio.construction import TopNConstructor
from clypt.strategy.base import SimpleStrategy


def load_universe_from_disk(universe_size: int = 10, exchange: str = "binance"):
    """
    Load pre-downloaded data from disk.

    Args:
        universe_size: Number of symbols to load
        exchange: Exchange name

    Returns:
        DataStore with loaded data
    """
    # Path to pre-downloaded data
    data_path = Path(__file__).parent.parent / "data" / exchange / "1d"

    if not data_path.exists():
        raise FileNotFoundError(
            f"No data found at {data_path}. "
            f"Run: python -m clypt.cli.data download"
        )

    # Get available symbols
    files = sorted(data_path.glob("*.parquet"))

    if len(files) < universe_size:
        raise ValueError(
            f"Only {len(files)} symbols available, but {universe_size} requested. "
            f"Run: python -m clypt.cli.data download --limit {universe_size}"
        )

    # Load data into DataStore
    store = DataStore()

    print(f"\nLoading {universe_size} symbols from {data_path}")
    print(f"Available: {len(files)} symbols")
    print("-" * 70)

    for i, filepath in enumerate(files[:universe_size], 1):
        # Convert filename to symbol (e.g., BTC_USDT.parquet -> BTC/USDT)
        symbol = filepath.stem.replace("_", "/")

        # Load parquet file
        df = pd.read_parquet(filepath)

        # Add to store
        store.add_ohlcv(symbol, df, frequency="1d", source=exchange)

        print(f"[{i:2d}/{universe_size}] {symbol:15s} {len(df)} bars")

    print("-" * 70)
    print(f"Loaded {universe_size} symbols\n")

    return store


class LargeUniverseStrategy(SimpleStrategy):
    """Strategy for large universes (10-50 symbols)."""

    def __init__(self, universe_size: int = 10):
        # Multiple factors for better signal
        factors = [
            MomentumFactor(lookback=20),
            VolatilityFactor(lookback=20),
        ]

        # Top N selection (e.g., top 20% of 50 symbols = top 10)
        top_n = max(3, universe_size // 5)

        # Position limits for large universes
        constraints = Constraints(
            max_position_size=0.3,  # Max 30% per position
            max_gross_exposure=1.0,
            min_position_size=0.05,  # Min 5% per position
            max_num_positions=top_n,
            allow_short=False,
        )

        super().__init__(
            factors_list=factors,
            constructor=TopNConstructor(top_n=top_n),
            constraints_obj=constraints,
            schedule_str="daily",
            warmup=25,
            name=f"Universe{universe_size}",
        )


def main():
    """Example: Backtest with pre-downloaded universe data."""

    # Configuration
    UNIVERSE_SIZE = 10  # Use 10 symbols (can be up to 50)
    BACKTEST_DAYS = 60

    # Load pre-downloaded data (very fast!)
    store = load_universe_from_disk(universe_size=UNIVERSE_SIZE)

    # Strategy with appropriate universe size
    strategy = LargeUniverseStrategy(universe_size=UNIVERSE_SIZE)

    # Setup engine
    cost_model = CostModel(
        maker_fee=0.001,
        taker_fee=0.001,
        slippage_bps=5.0,
    )

    executor = BacktestExecutor(cost_model)

    engine = Engine(
        strategy=strategy,
        data_store=store,
        mode=EngineMode.BACKTEST,
        executor=executor,
        initial_capital=10000.0,
    )

    # Run backtest on recent data
    date_range = store.get_date_range()
    end_date = date_range.end
    start_date = end_date - timedelta(days=BACKTEST_DAYS)

    print(f"Running backtest:")
    print(f"  Strategy:  {strategy.name}")
    print(f"  Universe:  {UNIVERSE_SIZE} symbols")
    print(f"  Period:    {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"  Days:      {BACKTEST_DAYS}")
    print()

    result = engine.run_backtest(start=start_date, end=end_date, verbose=True)

    # Print metrics
    print_metrics(result.metrics)

    print(f"\nBenefits of pre-downloaded data:")
    print(f"  - No API rate limits")
    print(f"  - Instant startup (no download time)")
    print(f"  - Offline development possible")
    print(f"  - Consistent data across runs")


if __name__ == "__main__":
    main()
