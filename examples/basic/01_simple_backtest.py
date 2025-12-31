"""
Example 1: Simple Backtest

Shows how to:
- Download live data from exchange
- Create a basic momentum strategy
- Run a backtest
- View results

Usage:
    python examples/01_simple_backtest.py
"""

from datetime import datetime, timedelta

from clyptq import Constraints, CostModel, EngineMode
from clyptq.analytics.metrics import print_metrics
from clyptq.data.loaders.ccxt import load_crypto_data
from clyptq.trading.engine import BacktestExecutor, Engine
from clyptq.trading.factors.library.momentum import MomentumFactor
from clyptq.trading.factors.library.volatility import VolatilityFactor
from clyptq.trading.portfolio.constructors import TopNConstructor
from clyptq.trading.strategy.base import SimpleStrategy


class MomentumStrategy(SimpleStrategy):
    """Buy high momentum, low volatility assets."""

    def __init__(self):
        factors = [
            MomentumFactor(lookback=20),
            VolatilityFactor(lookback=20),
        ]

        constraints = Constraints(
            max_position_size=0.4,
            max_gross_exposure=1.0,
            min_position_size=0.1,
            max_num_positions=3,
            allow_short=False,
        )

        super().__init__(
            factors_list=factors,
            constructor=TopNConstructor(top_n=3),
            constraints_obj=constraints,
            schedule_str="daily",
            warmup=25,
            name="Momentum",
        )


def main():
    # 1. Download data (live from exchange)
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT"]
    store = load_crypto_data(symbols=symbols, exchange="binance", timeframe="1d", days=180)

    # 2. Create strategy
    strategy = MomentumStrategy()

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

    # 4. Run backtest
    date_range = store.get_date_range()
    start_date = date_range.end - timedelta(days=90)
    end_date = date_range.end

    result = engine.run(start=start_date, end=end_date, verbose=True)

    # 5. View results
    print_metrics(result.metrics)


if __name__ == "__main__":
    main()
