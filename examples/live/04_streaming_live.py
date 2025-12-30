"""
Real-time trading with streaming data.

Demonstrates async streaming for minimal latency.
"""

import asyncio
import os

from clyptq.data.streams.ccxt_stream import CCXTStreamingSource
from clyptq.engine import Engine
from clyptq.execution.live import LiveExecutor
from clyptq.factors.library.momentum import MomentumFactor
from clyptq.portfolio.constructors import TopNConstructor
from clyptq.portfolio.constraints import Constraints
from clyptq.risk.costs import CostModel
from clyptq.risk.manager import RiskManager
from clyptq.strategy.base import SimpleStrategy
from clyptq.core.types import EngineMode


async def main():
    """Run streaming live trading."""

    # Strategy
    strategy = SimpleStrategy(
        factors_list=[MomentumFactor(lookback=20)],
        constructor=TopNConstructor(top_n=3),
        constraints_obj=Constraints(
            max_position_size=0.4,
            max_gross_exposure=1.0,
            min_position_size=0.05,
        ),
        schedule_str="daily",
        warmup=25,
        name="MomentumStream",
    )

    # Override universe for live trading
    def universe():
        return ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"]

    strategy.universe = universe

    # Executor
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")

    cost_model = CostModel(maker_fee=0.001, taker_fee=0.001, slippage_bps=5)
    executor = LiveExecutor(
        exchange_id="binance",
        api_key=api_key,
        api_secret=api_secret,
        cost_model=cost_model,
        paper_mode=True,  # Paper trading (no real orders)
        sandbox=False,
    )

    # Risk manager
    risk_manager = RiskManager(
        max_drawdown_pct=15.0,
        stop_loss_pct=5.0,
        take_profit_pct=10.0,
        max_position_size_pct=30.0,
    )

    # Engine
    engine = Engine(
        strategy=strategy,
        data_store=None,  # Not needed for live
        mode=EngineMode.PAPER,
        executor=executor,
        initial_capital=10000.0,
        risk_manager=risk_manager,
    )

    # Streaming source (1 sec updates)
    stream = CCXTStreamingSource(
        exchange_id="binance",
        api_key=api_key,
        api_secret=api_secret,
        poll_interval=1.0,
    )

    print("Starting streaming live trading...")
    print("Press Ctrl+C to stop\n")

    await engine.run_live_stream(stream, verbose=True)


if __name__ == "__main__":
    asyncio.run(main())
