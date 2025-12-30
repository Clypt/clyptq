"""
Multi-timeframe momentum strategy combining daily and weekly signals.

Demonstrates:
- MultiTimeframeStore with 1d and 1w data
- MultiTimeframeMomentum factor
- Daily rebalancing with weekly trend confirmation
"""

import pandas as pd
from datetime import datetime
from pathlib import Path

from clyptq.engine import Engine
from clyptq.execution.backtest import BacktestExecutor
from clyptq.data.mtf_store import MultiTimeframeStore
from clyptq.factors.library.momentum import MultiTimeframeMomentum
from clyptq.core.base import Factor
from clyptq.portfolio.construction import TopNConstructor
from clyptq.core.base import Strategy
from clyptq.core.types import EngineMode, Constraints, CostModel


class MultiTimeframeStrategy(Strategy):
    """Multi-timeframe momentum: 70% daily (20d) + 30% weekly (12w)."""

    def __init__(self):
        self.factor = MultiTimeframeMomentum(
            timeframes=["1d", "1w"],
            lookbacks={"1d": 20, "1w": 12},
            weights={"1d": 0.7, "1w": 0.3},
        )

    def factors(self):
        return [self.factor]

    def portfolio_constructor(self):
        return TopNConstructor(top_n=10)

    def constraints(self):
        return Constraints(
            max_position_size=0.15,
            min_position_size=0.05,
            max_gross_exposure=1.0,
            max_num_positions=10,
        )

    def schedule(self):
        return "daily"

    def warmup_periods(self):
        return 84  # 12 weeks


class MTFFactorAdapter(Factor):
    """Adapter for MultiTimeframeFactor to work with Engine."""

    def __init__(self, mtf_factor, mtf_store):
        super().__init__(mtf_factor.name)
        self.mtf_factor = mtf_factor
        self.mtf_store = mtf_store

    def compute(self, data):
        return self.mtf_factor.compute(
            self.mtf_store,
            data._timestamp,
            data.symbols,
        )


def load_top_symbols(data_dir: Path, top_n: int = 30) -> list:
    parquet_files = list(data_dir.glob("*.parquet"))
    volumes = []

    for file in parquet_files:
        try:
            df = pd.read_parquet(file)
            recent_volume = df.tail(30)["volume"].mean()
            volumes.append((file.stem, recent_volume))
        except Exception:
            continue

    volumes.sort(key=lambda x: x[1], reverse=True)
    return [sym for sym, _ in volumes[:top_n]]


def resample_to_weekly(daily_df: pd.DataFrame) -> pd.DataFrame:
    weekly = daily_df.resample('W').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    })
    return weekly.dropna()


def load_multi_timeframe_data(symbols: list, data_dir: Path) -> MultiTimeframeStore:
    mtf_store = MultiTimeframeStore()

    print("\nLoading multi-timeframe data:")
    for symbol in symbols:
        file_path = data_dir / f"{symbol}.parquet"
        if not file_path.exists():
            continue

        try:
            daily_df = pd.read_parquet(file_path)
            if not isinstance(daily_df.index, pd.DatetimeIndex):
                daily_df.index = pd.to_datetime(daily_df.index)

            mtf_store.add_ohlcv(symbol, daily_df, "1d")

            weekly_df = resample_to_weekly(daily_df)
            mtf_store.add_ohlcv(symbol, weekly_df, "1w")

            print(f"  ✓ {symbol}: {len(daily_df)} days, {len(weekly_df)} weeks")
        except Exception as e:
            print(f"  ✗ {symbol}: {e}")

    return mtf_store


def main():
    print("=" * 80)
    print("Multi-Timeframe Momentum Strategy")
    print("=" * 80)

    data_dir = Path("data/spot/binance/1d")

    print("\n[1/5] Loading top 30 symbols...")
    symbols = load_top_symbols(data_dir, top_n=30)
    print(f"Selected: {len(symbols)}")
    print(f"Top 10: {symbols[:10]}")

    print("\n[2/5] Loading multi-timeframe data...")
    mtf_store = load_multi_timeframe_data(symbols, data_dir)

    available_symbols = [
        s for s in symbols
        if "1d" in mtf_store.available_timeframes(s) and
           "1w" in mtf_store.available_timeframes(s)
    ]
    print(f"\nSymbols with 1d + 1w: {len(available_symbols)}")

    start = datetime(2023, 1, 1)
    end = datetime(2024, 12, 31)

    print(f"\n[3/5] Backtest: {start.date()} → {end.date()}")
    print(f"Duration: {(end - start).days} days")

    print("\n[4/5] Strategy configuration...")
    strategy = MultiTimeframeStrategy()
    mtf_factor = strategy.factors()[0]
    adapter = MTFFactorAdapter(mtf_factor, mtf_store)
    store_1d = mtf_store.get_store("1d")

    cost_model = CostModel(
        maker_fee=0.001,
        taker_fee=0.001,
        slippage_bps=5.0,
    )

    executor = BacktestExecutor(cost_model=cost_model)
    original_factors = strategy.factors
    strategy.factors = lambda: [adapter]

    engine = Engine(
        strategy=strategy,
        data_store=store_1d,
        mode=EngineMode.BACKTEST,
        executor=executor,
        initial_capital=100000.0,
    )

    print("Config: 1d(20) + 1w(12), weights 70%/30%, top 10, daily rebal")

    print("\n[5/5] Running backtest...")
    result = engine.run(start, end, verbose=True)
    strategy.factors = original_factors

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    m = result.metrics
    print(f"\nPerformance:")
    print(f"  Total Return:      {m.total_return:>10.2%}")
    print(f"  Annualized:        {m.annualized_return:>10.2%}")
    print(f"  Volatility:        {m.volatility:>10.2%}")
    print(f"  Sharpe:            {m.sharpe_ratio:>10.2f}")
    print(f"  Sortino:           {m.sortino_ratio:>10.2f}")
    print(f"  Max DD:            {m.max_drawdown:>10.2%}")

    print(f"\nTrading:")
    print(f"  Trades:            {m.num_trades:>10,}")
    print(f"  Win Rate:          {m.win_rate:>10.2%}")
    print(f"  Profit Factor:     {m.profit_factor:>10.2f}")

    if result.snapshots:
        final = result.snapshots[-5:]
        print(f"\nFinal Equity:")
        for snap in final:
            print(f"  {snap.timestamp.date()}: ${snap.equity:>12,.2f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
