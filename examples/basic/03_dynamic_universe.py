"""
Example 3: Dynamic Universe (NO Look-Ahead Bias)

Shows how to:
- Download ALL pairs to prevent look-ahead bias
- Select top N by volume dynamically at each rebalance
- Universe changes over time based on PAST data only

WRONG:
  Select top 50 by today's volume → use for entire backtest
  Problem: Coins popular now weren't popular 3 months ago!

RIGHT (this example):
  At each rebalance → select top 50 by PAST volume
  Universe changes as market evolves

Prerequisites:
    clypt-engine data download --all --days 90

Usage:
    python examples/03_dynamic_universe.py
"""

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from clyptq import Constraints, CostModel, EngineMode
from clyptq.analytics.metrics import print_metrics
from clyptq.data.stores.store import DataStore
from clyptq.trading.engine import Engine
from clyptq.trading.execution import BacktestExecutor
from clyptq.trading.factors.library.momentum import MomentumFactor
from clyptq.trading.factors.library.volatility import VolatilityFactor
from clyptq.trading.portfolio.constructors import TopNConstructor
from clyptq.trading.strategy.base import SimpleStrategy


def load_all_pairs(exchange: str = "binance", market: str = "spot") -> DataStore:
    """Load ALL pairs from disk."""
    data_path = Path(__file__).parent.parent / "data" / market / exchange / "1d"

    if not data_path.exists():
        raise FileNotFoundError(f"No data at {data_path}. Run: clypt-engine data download --all")

    store = DataStore()
    files = sorted(data_path.glob("*.parquet"))

    for filepath in files:
        symbol = filepath.stem.replace("_", "/")
        df = pd.read_parquet(filepath)
        store.add_ohlcv(symbol, df, frequency="1d", source=exchange)

    return store


class DynamicStrategy(SimpleStrategy):
    """Select top N by volume at each rebalance."""

    def __init__(self, universe_size: int = 50, positions: int = 10):
        self.universe_size = universe_size

        factors = [MomentumFactor(lookback=20), VolatilityFactor(lookback=20)]

        constraints = Constraints(
            max_position_size=0.25,
            max_gross_exposure=1.0,
            min_position_size=0.05,
            max_num_positions=positions,
            allow_short=False,
        )

        super().__init__(
            factors_list=factors,
            constructor=TopNConstructor(top_n=positions),
            constraints_obj=constraints,
            schedule_str="weekly",
            warmup=25,
            name=f"Dynamic{universe_size}",
        )


def main():
    # Config
    UNIVERSE_SIZE = 50
    POSITIONS = 10
    BACKTEST_DAYS = 60

    # 1. Load ALL pairs
    store = load_all_pairs()
    print(f"Loaded {len(store)} pairs\n")

    # 2. Strategy
    strategy = DynamicStrategy(universe_size=UNIVERSE_SIZE, positions=POSITIONS)

    # 3. Setup
    cost_model = CostModel(maker_fee=0.001, taker_fee=0.001, slippage_bps=5.0)
    executor = BacktestExecutor(cost_model)

    engine = Engine(
        strategy=strategy,
        data_store=store,
        mode=EngineMode.BACKTEST,
        executor=executor,
        initial_capital=10000.0,
    )

    # 4. Run
    date_range = store.get_date_range()
    start_date = date_range.end - timedelta(days=BACKTEST_DAYS)
    end_date = date_range.end

    print(f"Universe: Top {UNIVERSE_SIZE} by volume")
    print(f"Positions: Top {POSITIONS}")
    print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n")

    # Show universe changes over time
    test_dates = [start_date, start_date + timedelta(days=20), end_date]
    print("Universe evolution (top 5 by volume):")
    for dt in test_dates:
        top = store.get_top_symbols_by_volume(at=dt, top_n=5, lookback_days=7)
        print(f"  {dt.strftime('%Y-%m-%d')}: {', '.join(top)}")
    print()

    result = engine.run(start=start_date, end=end_date, verbose=True)

    # 5. Results
    print_metrics(result.metrics)


if __name__ == "__main__":
    main()
