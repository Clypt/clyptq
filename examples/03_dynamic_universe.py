"""Dynamic universe with NO look-ahead bias. Use --all flag when downloading."""

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


def load_full_universe(exchange: str = "binance", timeframe: str = "1d", market: str = "spot") -> DataStore:
    """Load all USDT pairs. This prevents look-ahead bias."""
    data_path = Path(__file__).parent.parent / "data" / market / exchange / timeframe

    if not data_path.exists():
        raise FileNotFoundError(
            f"No data found at {data_path}.\n"
            f"Run: python -m clypt.cli.data download --all --days 90"
        )

    files = sorted(data_path.glob("*.parquet"))

    if len(files) == 0:
        raise ValueError(
            "No data files found.\n"
            "Run: python -m clypt.cli.data download --all --days 90"
        )

    store = DataStore()

    print(f"\nLoading universe data from {data_path}")
    print(f"Found {len(files)} USDT pairs")
    print("-" * 70)

    loaded = 0
    for filepath in files:
        symbol = filepath.stem.replace("_", "/")

        try:
            df = pd.read_parquet(filepath)
            store.add_ohlcv(symbol, df, frequency=timeframe, source=exchange)
            loaded += 1
        except Exception as e:
            print(f"Warning: Failed to load {symbol}: {e}")
            continue

    print("-" * 70)
    print(f"Successfully loaded {loaded}/{len(files)} symbols")
    print(f"Date range: {store.get_date_range()}")
    print()

    return store


class DynamicUniverseStrategy(SimpleStrategy):
    """Dynamic universe - top N by past volume only."""

    def __init__(
        self,
        universe_size: int = 50,
        top_n_positions: int = 10,
        volume_lookback_days: int = 7,
    ):
        """Init with dynamic selection."""
        self.universe_size = universe_size
        self.volume_lookback_days = volume_lookback_days

        factors = [MomentumFactor(lookback=20), VolatilityFactor(lookback=20)]

        constraints = Constraints(
            max_position_size=0.25,
            max_gross_exposure=1.0,
            min_position_size=0.05,
            max_num_positions=top_n_positions,
            allow_short=False,
        )

        super().__init__(
            factors_list=factors,
            constructor=TopNConstructor(top_n=top_n_positions),
            constraints_obj=constraints,
            schedule_str="weekly",
            warmup=25,
            name=f"Dynamic{universe_size}",
        )


def main():
    """Backtest with dynamic universe."""
    UNIVERSE_SIZE = 50
    TOP_N_POSITIONS = 10
    VOLUME_LOOKBACK = 7
    BACKTEST_DAYS = 60

    print("=" * 70)
    print("DYNAMIC UNIVERSE BACKTESTING (NO LOOK-AHEAD BIAS)")
    print("=" * 70)

    store = load_full_universe()

    strategy = DynamicUniverseStrategy(
        universe_size=UNIVERSE_SIZE,
        top_n_positions=TOP_N_POSITIONS,
        volume_lookback_days=VOLUME_LOOKBACK,
    )

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

    print(f"\nBacktest Configuration:")
    print(f"  Strategy:         {strategy.name}")
    print(f"  Universe Size:    Top {UNIVERSE_SIZE} by volume")
    print(f"  Positions:        Hold top {TOP_N_POSITIONS}")
    print(f"  Volume Lookback:  {VOLUME_LOOKBACK} days")
    print(f"  Rebalancing:      {strategy.schedule()}")
    print(f"  Period:           {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"  Days:             {BACKTEST_DAYS}")
    print()

    # Demonstrate dynamic universe selection
    print("Example: Universe selection at different dates")
    print("-" * 70)

    test_dates = [
        start_date,
        start_date + timedelta(days=15),
        start_date + timedelta(days=30),
        end_date,
    ]

    for test_date in test_dates:
        top_symbols = store.get_top_symbols_by_volume(
            at=test_date, top_n=10, lookback_days=VOLUME_LOOKBACK
        )
        print(
            f"{test_date.strftime('%Y-%m-%d')}: {', '.join(top_symbols[:5])}..."
        )

    print("-" * 70)
    print("\nNotice: Universe changes over time based on PAST volume data")
    print("This prevents look-ahead bias!\n")

    print("Running backtest...")
    result = engine.run_backtest(start=start_date, end=end_date, verbose=True)

    print_metrics(result.metrics)

    print("\n" + "=" * 70)
    print("WHY THIS PREVENTS LOOK-AHEAD BIAS:")
    print("=" * 70)
    print("""
1. We downloaded ALL USDT pairs (~640 symbols)
2. At each rebalancing date, we select top 50 by PAST volume only
3. Universe membership changes as market conditions evolve
4. We NEVER use future information to select symbols

WRONG Approach (Look-Ahead Bias):
  - Select top 50 symbols by 2025-12-28 volume
  - Use those 50 for entire backtest from Oct-Dec
  - Problem: Some coins in top 50 now weren't popular in October!

RIGHT Approach (This Example):
  - Oct 1: Select top 50 by Sep volume
  - Nov 1: Select top 50 by Oct volume
  - Dec 1: Select top 50 by Nov volume
  - Each selection uses ONLY past data!

Result: Realistic backtest that could be replicated in live trading.
""")


if __name__ == "__main__":
    main()
