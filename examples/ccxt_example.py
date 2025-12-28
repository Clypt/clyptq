"""
CCXTExecutor example for paper and live trading.
Shows unified interface for both modes.
"""

from datetime import timedelta

from clypt import Constraints, CostModel, EngineMode
from clypt.analytics.metrics import print_metrics
from clypt.data.loaders.ccxt import load_crypto_data
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
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
    store = load_crypto_data(symbols, days=90)

    strategy = SimpleStrategy()

    executor = CCXTExecutor(
        exchange_id="binance",
        api_key="test",
        api_secret="test",
        paper_mode=True,
        cost_model=CostModel(),
    )

    engine = Engine(
        strategy=strategy,
        data_store=store,
        mode=EngineMode.BACKTEST,
        executor=executor,
        initial_capital=10000.0,
    )

    date_range = store.get_date_range()
    end_date = date_range.end
    start_date = end_date - timedelta(days=30)

    result = engine.run_backtest(start=start_date, end=end_date, verbose=True)

    print_metrics(result.metrics)

    data = result.to_dict()
    print(f"\nExported: {len(data['equity_curve'])} points, {len(data['trades'])} trades")

    # For live: paper_mode=False, sandbox=True


if __name__ == "__main__":
    main()
