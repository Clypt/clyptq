"""
Professional backtest with advanced factor combining momentum and mean reversion.

Uses operations library for sophisticated factor construction:
- Momentum score with trend strength
- Mean reversion with volatility adjustment
- Cross-sectional ranking
"""

import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from clyptq.trading.engine import Engine
from clyptq.trading.execution.backtest import BacktestExecutor
from clyptq.data.stores.store import DataStore
from clyptq.core.base import Factor
from clyptq.trading.factors.ops.time_series import ts_mean, ts_std, ts_rank, correlation
from clyptq.trading.factors.ops.cross_sectional import rank
from clyptq.trading.portfolio.constructors import TopNConstructor
from clyptq.trading.strategy.base import SimpleStrategy
from clyptq.core.types import EngineMode, Constraints, CostModel


class MomentumMeanReversionFactor(Factor):
    """
    Advanced factor combining momentum and mean reversion signals.

    Strategy:
    1. Momentum Score: Price trend strength (correlation with linear trend)
    2. Mean Reversion Score: Z-score normalized by volatility
    3. Combined Signal: Weighted combination with cross-sectional ranking

    Logic:
    - Strong momentum + not overbought = BUY
    - Weak momentum + oversold = potential reversal
    """

    def __init__(
        self,
        momentum_lookback: int = 60,
        zscore_lookback: int = 20,
        momentum_weight: float = 0.6,
        name: str = "MomentumMeanReversion",
    ):
        super().__init__(name)
        self.momentum_lookback = momentum_lookback
        self.zscore_lookback = zscore_lookback
        self.momentum_weight = momentum_weight
        self.mean_rev_weight = 1.0 - momentum_weight

    def compute(self, data):
        scores = {}

        for symbol in data.symbols:
            try:
                # Get required lookback data
                lookback = max(self.momentum_lookback, self.zscore_lookback)
                close = data.close(symbol, lookback)

                # Momentum signal: correlation with linear trend
                linear_trend = np.arange(self.momentum_lookback)
                momentum_score = correlation(close, linear_trend, self.momentum_lookback)

                if np.isnan(momentum_score):
                    continue

                # Mean reversion signal: z-score
                mean = ts_mean(close, self.zscore_lookback)
                std = ts_std(close, self.zscore_lookback)

                if np.isnan(mean) or np.isnan(std) or std < 1e-8:
                    continue

                zscore = (close[-1] - mean) / std
                # Invert z-score: negative when overbought, positive when oversold
                mean_rev_score = -zscore

                # Combined score with weights
                combined = (
                    self.momentum_weight * momentum_score +
                    self.mean_rev_weight * mean_rev_score
                )

                scores[symbol] = combined

            except (KeyError, ValueError):
                # Insufficient data for this symbol
                continue

        # Cross-sectional ranking for final scores
        if scores:
            ranked = rank(scores)
            # Debug: print first time
            if not hasattr(self, '_debug_printed'):
                print(f"  Factor computed {len(scores)} raw scores → {len(ranked)} ranked scores")
                if ranked:
                    sample_items = list(ranked.items())[:3]
                    for sym, score in sample_items:
                        print(f"    {sym}: {score:.4f}")
                self._debug_printed = True
            return ranked

        return {}


def load_top_symbols(data_dir: Path, top_n: int = 50) -> list:
    """Load top N symbols by recent volume."""
    parquet_files = list(data_dir.glob("*.parquet"))

    volumes = []
    for file in parquet_files:
        try:
            df = pd.read_parquet(file)
            # Get average volume from last 30 days
            recent_volume = df.tail(30)["volume"].mean()
            symbol = file.stem  # BTC_USDT
            volumes.append((symbol, recent_volume))
        except Exception:
            continue

    # Sort by volume and take top N
    volumes.sort(key=lambda x: x[1], reverse=True)
    top_symbols = [sym for sym, _ in volumes[:top_n]]

    return top_symbols


def load_data_from_parquet(symbols: list, data_dir: Path) -> DataStore:
    """Load parquet data into DataStore."""
    store = DataStore()

    for symbol in symbols:
        file_path = data_dir / f"{symbol}.parquet"
        if not file_path.exists():
            continue

        try:
            df = pd.read_parquet(file_path)
            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            store.add_ohlcv(symbol, df)
            print(f"✓ {symbol}: {len(df)} days")
        except Exception as e:
            print(f"✗ {symbol}: {e}")

    return store


def main():
    print("=" * 80)
    print("Professional Backtest: Momentum + Mean Reversion Factor")
    print("=" * 80)

    # Data directory
    data_dir = Path("data/spot/binance/1d")

    # Load top 50 symbols by volume
    print("\n[1/5] Loading top 50 symbols by volume...")
    symbols = load_top_symbols(data_dir, top_n=50)
    print(f"Selected symbols: {len(symbols)}")
    print(f"Top 10: {symbols[:10]}")

    # Load data
    print("\n[2/5] Loading OHLCV data...")
    store = load_data_from_parquet(symbols, data_dir)
    print(f"Loaded: {len(store.symbols())} symbols")

    # Backtest period: 3 years (2022-01-01 to 2024-12-31)
    start = datetime(2022, 1, 1)
    end = datetime(2024, 12, 31)

    print(f"\n[3/5] Backtest period: {start.date()} → {end.date()}")
    print(f"Duration: {(end - start).days} days")

    # Strategy configuration
    print("\n[4/5] Strategy configuration...")
    factor = MomentumMeanReversionFactor(
        momentum_lookback=60,
        zscore_lookback=20,
        momentum_weight=0.6,  # 60% momentum, 40% mean reversion
    )

    strategy = SimpleStrategy(
        factors_list=[factor],
        constructor=TopNConstructor(top_n=10),
        constraints_obj=Constraints(
            max_position_size=0.15,  # Max 15% per position
            min_position_size=0.05,  # Min 5% per position
            max_gross_exposure=1.0,  # Fully invested
            max_num_positions=10,
        ),
        warmup=60,  # Need 60 days for momentum calculation
        schedule_str="weekly",  # Rebalance weekly
    )

    cost_model = CostModel(
        maker_fee=0.001,  # 0.1% maker fee
        taker_fee=0.001,  # 0.1% taker fee
        slippage_bps=5.0,  # 5 bps slippage
    )

    executor = BacktestExecutor(cost_model=cost_model)

    engine = Engine(
        strategy=strategy,
        data_store=store,
        mode=EngineMode.BACKTEST,
        executor=executor,
        initial_capital=100000.0,
    )

    print("Strategy:")
    print(f"  Factor: MomentumMeanReversion (60d momentum, 20d zscore)")
    print(f"  Weight: 60% momentum + 40% mean reversion")
    print(f"  Portfolio: Top 10 positions, weekly rebalance")
    print(f"  Position limits: 5-15% per asset")
    print(f"  Costs: 0.1% fees, 5bps slippage")

    # Run backtest
    print("\n[5/5] Running backtest...")
    result = engine.run(start, end, verbose=True)

    # Results
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)

    metrics = result.metrics

    print(f"\nPerformance Metrics:")
    print(f"  Total Return:      {metrics.total_return:>10.2%}")
    print(f"  Annualized Return: {metrics.annualized_return:>10.2%}")
    print(f"  Volatility:        {metrics.volatility:>10.2%}")
    print(f"  Sharpe Ratio:      {metrics.sharpe_ratio:>10.2f}")
    print(f"  Sortino Ratio:     {metrics.sortino_ratio:>10.2f}")
    print(f"  Max Drawdown:      {metrics.max_drawdown:>10.2%}")

    print(f"\nTrading Activity:")
    print(f"  Total Trades:      {metrics.num_trades:>10,}")
    print(f"  Win Rate:          {metrics.win_rate:>10.2%}")
    print(f"  Profit Factor:     {metrics.profit_factor:>10.2f}")
    print(f"  Avg Trade P&L:     ${metrics.avg_trade_pnl:>10,.2f}")

    print(f"\nRisk Metrics:")
    print(f"  Avg Leverage:      {metrics.avg_leverage:>10.2f}x")
    print(f"  Max Leverage:      {metrics.max_leverage:>10.2f}x")
    print(f"  Avg Positions:     {metrics.avg_num_positions:>10.1f}")

    print(f"\nPeriod:")
    print(f"  Start Date:        {metrics.start_date.date()}")
    print(f"  End Date:          {metrics.end_date.date()}")
    print(f"  Duration:          {metrics.duration_days} days")

    # Final equity curve
    if result.snapshots:
        final_snapshots = result.snapshots[-10:]
        print(f"\nFinal Equity Curve (last 10 snapshots):")
        for snap in final_snapshots:
            print(f"  {snap.timestamp.date()}: ${snap.equity:>12,.2f}")
    else:
        print(f"\n⚠ No snapshots generated")

    # Export to JSON
    result_dict = result.to_dict()
    import json
    with open("backtest_result.json", "w") as f:
        json.dump(result_dict, f, indent=2)
    print(f"\n✓ Results exported to backtest_result.json")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
