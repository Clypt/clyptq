"""
CCXTExecutor example for paper and live trading.
Shows unified interface for both modes.
"""

from datetime import timedelta

from clypt import Constraints, CostModel, EngineMode
from clypt.analytics.metrics import print_metrics
from clypt.data.loaders.ccxt_loader import load_crypto_data
from clypt.engine import CCXTExecutor, Engine
from clypt.factors.library.momentum import MomentumFactor
from clypt.portfolio.construction import TopNConstructor
from clypt.strategy.base import Strategy


class SimpleStrategy(Strategy):
    def __init__(self):
        super().__init__("Simple")

    def factors(self):
        return [MomentumFactor(lookback=14)]

    def portfolio_constructor(self):
        return TopNConstructor(top_n=3)

    def constraints(self):
        return Constraints(max_position_size=0.4)

    def schedule(self):
        return "daily"

    def warmup_periods(self):
        return 20


def main():
    print("=" * 60)
    print("CCXT Executor Example")
    print("=" * 60)

    # Load data
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
    store = load_crypto_data(symbols, days=90)

    # Create strategy
    strategy = SimpleStrategy()

    # Paper mode executor
    executor = CCXTExecutor(
        exchange_id="binance",
        api_key="test",  # not used in paper mode
        api_secret="test",
        paper_mode=True,  # simulation only
        cost_model=CostModel(),
    )

    engine = Engine(
        strategy=strategy,
        data_store=store,
        mode=EngineMode.BACKTEST,
        executor=executor,
        initial_capital=10000.0,
    )

    # Run backtest
    date_range = store.get_date_range()
    end_date = date_range.end
    start_date = end_date - timedelta(days=30)

    print("\nRunning paper mode backtest...")
    result = engine.run_backtest(start=start_date, end=end_date, verbose=True)

    print_metrics(result.metrics)

    # Export to SaaS format
    print("\nExporting to SaaS format...")
    data = result.to_dict()
    print(f"✅ Equity curve: {len(data['equity_curve'])} points")
    print(f"✅ Trades: {len(data['trades'])}")
    print(f"✅ Metrics: {len(data['metrics'])} fields")

    print("\n" + "=" * 60)
    print("Paper mode test complete!")
    print("=" * 60)

    # For live trading, just change paper_mode=False
    # executor = CCXTExecutor(
    #     exchange_id="binance",
    #     api_key="YOUR_REAL_KEY",
    #     api_secret="YOUR_REAL_SECRET",
    #     paper_mode=False,
    #     sandbox=True  # use testnet first!
    # )


if __name__ == "__main__":
    main()
