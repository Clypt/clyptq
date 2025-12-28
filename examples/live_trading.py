"""
Live trading example with CCXTExecutor.
Run this for paper or live mode real-time trading.
"""

from typing import List

from clypt import Constraints, CostModel, EngineMode
from clypt.data.store import DataStore
from clypt.engine import CCXTExecutor, Engine, RiskManager
from clypt.factors.base import Factor
from clypt.factors.library.momentum import MomentumFactor
from clypt.portfolio.construction import TopNConstructor
from clypt.strategy.base import Strategy


class LiveStrategy(Strategy):
    """Simple strategy for live trading."""

    def __init__(self):
        super().__init__("LiveMomentum")
        self._universe = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]

    def factors(self) -> List[Factor]:
        return [MomentumFactor(lookback=14)]

    def portfolio_constructor(self):
        return TopNConstructor(top_n=2)

    def constraints(self):
        return Constraints(max_position_size=0.5)

    def schedule(self):
        return "daily"

    def universe(self):
        return self._universe

    def warmup_periods(self):
        return 0  # no warmup for live


def main():
    print("=" * 60)
    print("Live Trading Example")
    print("=" * 60)

    # Paper mode (safe for testing)
    executor = CCXTExecutor(
        exchange_id="binance",
        api_key="dummy_key",  # not used in paper mode
        api_secret="dummy_secret",
        paper_mode=True,  # SIMULATION ONLY
        cost_model=CostModel(),
    )

    strategy = LiveStrategy()
    data_store = DataStore()  # empty store for live mode

    # Risk management (optional)
    risk_manager = RiskManager(
        stop_loss_pct=0.05,      # 5% stop loss
        take_profit_pct=0.10,     # 10% take profit
        max_drawdown_pct=0.15     # 15% max drawdown kill switch
    )

    engine = Engine(
        strategy=strategy,
        data_store=data_store,
        mode=EngineMode.PAPER,
        executor=executor,
        initial_capital=10000.0,
        risk_manager=risk_manager,
    )

    print("\n🚀 Starting paper trading...")
    print("Press Ctrl+C to stop\n")

    # Run live trading loop
    # Fetches prices every 60 seconds and rebalances based on strategy schedule
    engine.run_live(interval_seconds=60, verbose=True)


if __name__ == "__main__":
    # For REAL money trading, change to:
    # executor = CCXTExecutor(
    #     exchange_id="binance",
    #     api_key="YOUR_REAL_API_KEY",
    #     api_secret="YOUR_REAL_SECRET",
    #     paper_mode=False,  # REAL MONEY
    #     sandbox=True,      # USE TESTNET FIRST!
    # )
    # engine = Engine(..., mode=EngineMode.LIVE, executor=executor)
    main()
