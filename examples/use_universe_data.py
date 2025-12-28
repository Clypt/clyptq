"""Use pre-downloaded universe data (fast, no API calls)."""

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
    """Load pre-downloaded data. Run 'python -m clypt.cli.data download' first."""
    data_path = Path(__file__).parent.parent / "data" / exchange / "1d"

    if not data_path.exists():
        raise FileNotFoundError(f"No data at {data_path}. Run: python -m clypt.cli.data download")

    files = sorted(data_path.glob("*.parquet"))

    if len(files) < universe_size:
        raise ValueError(f"Only {len(files)} symbols, need {universe_size}")

    store = DataStore()

    print(f"\nLoading {universe_size} symbols")
    for i, filepath in enumerate(files[:universe_size], 1):
        symbol = filepath.stem.replace("_", "/")
        df = pd.read_parquet(filepath)
        store.add_ohlcv(symbol, df, frequency="1d", source=exchange)
        print(f"[{i:2d}/{universe_size}] {symbol:15s} {len(df)} bars")

    print(f"Loaded {universe_size} symbols\n")
    return store


class LargeUniverseStrategy(SimpleStrategy):
    def __init__(self, universe_size: int = 10):
        factors = [
            MomentumFactor(lookback=20),
            VolatilityFactor(lookback=20),
        ]

        top_n = max(3, universe_size // 5)

        constraints = Constraints(
            max_position_size=0.3,
            max_gross_exposure=1.0,
            min_position_size=0.05,
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
    UNIVERSE_SIZE = 10
    BACKTEST_DAYS = 60

    store = load_universe_from_disk(universe_size=UNIVERSE_SIZE)
    strategy = LargeUniverseStrategy(universe_size=UNIVERSE_SIZE)

    cost_model = CostModel(maker_fee=0.001, taker_fee=0.001, slippage_bps=5.0)
    executor = BacktestExecutor(cost_model)

    engine = Engine(
        strategy=strategy,
        data_store=store,
        mode=EngineMode.BACKTEST,
        executor=executor,
        initial_capital=10000.0,
    )

    date_range = store.get_date_range()
    end_date = date_range.end
    start_date = end_date - timedelta(days=BACKTEST_DAYS)

    result = engine.run_backtest(start=start_date, end=end_date, verbose=True)
    print_metrics(result.metrics)

    print(f"\nBenefits: instant startup, offline dev, consistent data")


if __name__ == "__main__":
    main()
