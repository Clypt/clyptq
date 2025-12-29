"""
Real strategy backtest with HTML report.

Downloads real data, runs momentum strategy, generates report.
"""

import pandas as pd
from datetime import datetime
from pathlib import Path

from clyptq.engine.core import Engine
from clyptq.execution.backtest import BacktestExecutor
from clyptq.data.store import DataStore
from clyptq.factors.library.momentum import MomentumFactor
from clyptq.portfolio.construction import TopNConstructor
from clyptq.strategy.base import Strategy
from clyptq.core.types import EngineMode, Constraints, CostModel
from clyptq.analytics.report import HTMLReportGenerator


class SimpleMomentumStrategy(Strategy):
    """Simple 20-day momentum strategy."""

    def __init__(self):
        self.name = "Momentum20D"
        self.momentum = MomentumFactor(lookback=20)

    def factors(self):
        return [self.momentum]

    def portfolio_constructor(self):
        return TopNConstructor(top_n=5)

    def constraints(self):
        return Constraints(
            max_position_size=0.25,
            min_position_size=0.10,
            max_gross_exposure=1.0,
            max_num_positions=5,
        )

    def schedule(self):
        return "daily"

    def warmup_periods(self):
        return 25


def get_major_symbols() -> list:
    """Get major crypto symbols."""
    return [
        "BTC/USDT",
        "ETH/USDT",
        "BNB/USDT",
        "SOL/USDT",
        "XRP/USDT",
        "ADA/USDT",
        "DOGE/USDT",
        "DOT/USDT",
        "MATIC/USDT",
        "AVAX/USDT",
        "LINK/USDT",
        "UNI/USDT",
        "ATOM/USDT",
        "LTC/USDT",
        "ETC/USDT",
        "FIL/USDT",
        "AAVE/USDT",
        "ALGO/USDT",
        "SAND/USDT",
        "MANA/USDT",
    ]


def load_data(symbols: list, data_dir: Path) -> DataStore:
    """Load OHLCV data into DataStore."""
    store = DataStore()

    print("\nLoading data:")
    for symbol in symbols:
        file_name = symbol.replace("/", "_")
        file_path = data_dir / f"{file_name}.parquet"
        if not file_path.exists():
            continue

        try:
            df = pd.read_parquet(file_path)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            store.add_ohlcv(symbol, df)
            print(f"  ✓ {symbol}: {len(df)} bars")
        except Exception as e:
            print(f"  ✗ {symbol}: {e}")

    return store


def main():
    print("=" * 80)
    print("Real Strategy Backtest with HTML Report")
    print("=" * 80)

    data_dir = Path("data/spot/binance/1d")

    if not data_dir.exists():
        print(f"\nError: Data directory not found: {data_dir}")
        print("Please run: clyptq data download --exchange binance --days 365 --limit 20")
        return

    print("\n[1/6] Loading major symbols...")
    symbols = get_major_symbols()
    print(f"Selected: {len(symbols)} symbols")
    print(f"Symbols: {', '.join(symbols[:10])}")

    print("\n[2/6] Loading OHLCV data...")
    store = load_data(symbols, data_dir)
    print(f"Total symbols loaded: {len(store.symbols())}")

    start = datetime(2024, 1, 1)
    end = datetime(2024, 12, 25)

    print(f"\n[3/6] Backtest period: {start.date()} → {end.date()}")
    print(f"Duration: {(end - start).days} days")

    print("\n[4/6] Running backtest...")
    strategy = SimpleMomentumStrategy()

    cost_model = CostModel(
        maker_fee=0.001,
        taker_fee=0.001,
        slippage_bps=5.0,
    )

    executor = BacktestExecutor(cost_model=cost_model)

    engine = Engine(
        strategy=strategy,
        data_store=store,
        mode=EngineMode.BACKTEST,
        executor=executor,
        initial_capital=100000.0,
    )

    result = engine.run_backtest(start, end, verbose=True)

    print("\n[5/6] Backtest Results:")
    m = result.metrics
    print(f"  Total Return:      {m.total_return:>10.2%}")
    print(f"  Annualized:        {m.annualized_return:>10.2%}")
    print(f"  Sharpe Ratio:      {m.sharpe_ratio:>10.2f}")
    print(f"  Max Drawdown:      {m.max_drawdown:>10.2%}")
    print(f"  Trades:            {m.num_trades:>10,}")
    print(f"  Win Rate:          {m.win_rate:>10.2%}")

    print("\n[6/6] Generating HTML report...")
    generator = HTMLReportGenerator(rolling_window=30, min_drawdown=0.01)

    output_path = "momentum_strategy_report.html"
    generator.generate(
        result=result,
        output_path=output_path,
        title="20-Day Momentum Strategy Report",
    )

    print(f"  ✓ Report saved to: {output_path}")

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"\nOpen report: open {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
