"""
Example 2: Pre-downloaded Universe Data

Shows how to:
- Use pre-downloaded data (faster, no API calls)
- Load from disk
- Run backtests offline

Prerequisites:
    clypt-engine data download --symbols BTC/USDT ETH/USDT BNB/USDT --days 90

Usage:
    python examples/02_universe_data.py
"""

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from clyptq import Constraints, CostModel, EngineMode
from clyptq.analytics.metrics import print_metrics
from clyptq.data.stores.store import DataStore
from clyptq.engine import Engine
from clyptq.execution import BacktestExecutor
from clyptq.factors.library.momentum import MomentumFactor
from clyptq.factors.library.volatility import VolatilityFactor
from clyptq.portfolio.constructors import TopNConstructor
from clyptq.strategy.base import SimpleStrategy


def load_from_disk(n: int = 10, exchange: str = "binance", market: str = "spot") -> DataStore:
    """Load pre-downloaded data from disk."""
    data_path = Path(__file__).parent.parent / "data" / market / exchange / "1d"

    if not data_path.exists():
        raise FileNotFoundError(f"No data at {data_path}. Run: clypt-engine data download")

    files = sorted(data_path.glob("*.parquet"))[:n]
    if not files:
        raise ValueError(f"No data files found")

    store = DataStore()
    for filepath in files:
        symbol = filepath.stem.replace("_", "/")
        df = pd.read_parquet(filepath)
        store.add_ohlcv(symbol, df, frequency="1d", source=exchange)

    return store


class BasicStrategy(SimpleStrategy):
    """Basic momentum + volatility strategy."""

    def __init__(self):
        factors = [MomentumFactor(lookback=20), VolatilityFactor(lookback=20)]

        constraints = Constraints(
            max_position_size=0.3,
            max_gross_exposure=1.0,
            min_position_size=0.05,
            max_num_positions=3,
            allow_short=False,
        )

        super().__init__(
            factors_list=factors,
            constructor=TopNConstructor(top_n=3),
            constraints_obj=constraints,
            schedule_str="daily",
            warmup=25,
            name="Basic",
        )


def main():
    # 1. Load from disk (fast!)
    store = load_from_disk(n=10)
    print(f"Loaded {len(store)} symbols")

    # 2. Create strategy
    strategy = BasicStrategy()

    # 3. Setup backtest
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
    start_date = date_range.end - timedelta(days=60)
    end_date = date_range.end

    result = engine.run(start=start_date, end=end_date, verbose=True)

    # 5. Results
    print_metrics(result.metrics)


if __name__ == "__main__":
    main()
