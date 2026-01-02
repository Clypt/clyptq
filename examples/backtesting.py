"""
Backtesting Example

Usage:
    python examples/backtesting.py
"""

from datetime import timedelta

from clyptq import CostModel
from clyptq.analytics.performance.metrics import print_metrics
from clyptq.data.loaders.ccxt import load_crypto_data
from clyptq.trading.engine import BacktestEngine
from clyptq.trading.execution import BacktestExecutor

from strategy import MomentumStrategy


def main():
    print("=" * 70)
    print("BACKTEST MODE")
    print("=" * 70)

    # 1. Download data
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT"]
    print(f"\nDownloading {len(symbols)} symbols...")
    store = load_crypto_data(symbols=symbols, exchange="binance", timeframe="1d", days=720)
    print("Data loaded")

    # 2. Setup
    strategy = MomentumStrategy()
    cost_model = CostModel(maker_fee=0.001, taker_fee=0.001, slippage_bps=5.0)
    executor = BacktestExecutor(cost_model)

    engine = BacktestEngine(
        strategy=strategy,
        data_store=store,
        executor=executor,
        initial_capital=10000.0,
    )

    # 3. Run
    date_range = store.get_date_range()
    start = date_range.end - timedelta(days=720)
    end = date_range.end

    print(f"\nBacktest period: {start.date()} to {end.date()}")
    print(f"Strategy: {strategy.name}")
    print(f"Initial capital: $10,000\n")

    result = engine.run(start=start, end=end, verbose=True)

    # 4. Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print_metrics(result.metrics)


if __name__ == "__main__":
    main()
