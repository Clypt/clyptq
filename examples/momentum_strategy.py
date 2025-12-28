"""Example momentum strategy."""

from datetime import datetime, timedelta
from typing import List

from clypt import Constraints, CostModel, EngineMode
from clypt.analytics.metrics import print_metrics
from clypt.data.loaders.ccxt import load_crypto_data
from clypt.engine import BacktestExecutor, Engine
from clypt.factors.base import Factor, combine_factors
from clypt.factors.library.momentum import MomentumFactor, RSIFactor
from clypt.portfolio.construction import PortfolioConstructor, TopNConstructor
from clypt.strategy.base import Strategy


class MomentumStrategy(Strategy):
    def __init__(
        self,
        momentum_lookback: int = 20,
        rsi_lookback: int = 14,
        top_n: int = 5,
        max_position_size: float = 0.3,
        name: str = "Momentum",
    ):
        super().__init__(name)
        self.momentum_lookback = momentum_lookback
        self.rsi_lookback = rsi_lookback
        self.top_n = top_n
        self.max_position_size = max_position_size

    def factors(self) -> List[Factor]:
        momentum = MomentumFactor(lookback=self.momentum_lookback, name="Momentum20")
        rsi = RSIFactor(lookback=self.rsi_lookback, name="RSI14")
        combined = combine_factors([momentum, rsi], weights=[0.7, 0.3], name="Combined")
        return [combined]

    def portfolio_constructor(self) -> PortfolioConstructor:
        return TopNConstructor(top_n=self.top_n)

    def constraints(self) -> Constraints:
        return Constraints(
            max_position_size=self.max_position_size,
            max_gross_exposure=1.0,
            min_position_size=0.05,
            max_num_positions=self.top_n,
            allow_short=False,
        )

    def schedule(self) -> str:
        return "daily"

    def warmup_periods(self) -> int:
        return max(self.momentum_lookback, self.rsi_lookback) + 10


def main():
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT"]

    store = load_crypto_data(
        symbols=symbols,
        exchange="binance",
        timeframe="1d",
        days=180,
    )

    strategy = MomentumStrategy(
        momentum_lookback=20,
        rsi_lookback=14,
        top_n=3,
        max_position_size=0.4,
    )

    cost_model = CostModel(
        maker_fee=0.001,
        taker_fee=0.001,
        slippage_bps=5.0,
    )

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
    start_date = end_date - timedelta(days=90)

    result = engine.run_backtest(
        start=start_date, end=end_date, verbose=True
    )

    print_metrics(result.metrics)

    print(f"\nTrades: {len(result.trades)}")

    if result.snapshots:
        final = result.snapshots[-1]
        print(f"\nFinal: ${final.equity:.2f} ({final.num_positions} positions)")

        if final.positions:
            for symbol, pos in final.positions.items():
                print(f"  {symbol}: {pos.amount:.4f} @ ${pos.avg_price:.2f}")


if __name__ == "__main__":
    main()
