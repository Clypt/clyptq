"""
Example 5: Paper Trading with Real Prices

Shows how to:
- Run paper trading with live market prices
- Use the same strategy as backtesting
- Monitor real-time rebalancing
- Track P&L without real money

Usage:
    python examples/05_paper_trading.py

Note: Press Ctrl+C to stop and view results
"""

import os
import time
from datetime import datetime

from clyptq import Constraints, CostModel, EngineMode
from clyptq.data.stores.live_store import LiveDataStore
from clyptq.engine import Engine
from clyptq.execution.live import CCXTExecutor
from clyptq.factors.library.momentum import MomentumFactor
from clyptq.factors.library.volatility import VolatilityFactor
from clyptq.portfolio.constructors import TopNConstructor
from clyptq.strategy.base import SimpleStrategy


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
    print("=" * 70)
    print("PAPER TRADING MODE - Live Prices, No Real Orders")
    print("=" * 70)

    # Strategy
    strategy = MomentumStrategy()
    universe = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT"]

    print(f"\nStrategy: {strategy.name}")
    print(f"Universe: {', '.join(universe)}")
    print(f"Schedule: {strategy.schedule()}")
    print(f"Warmup: {strategy.warmup_periods()} periods")

    initial_capital = 10000.0
    print(f"Capital: ${initial_capital:,.0f}\n")

    # Setup executor (paper mode)
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")

    cost_model = CostModel(maker_fee=0.001, taker_fee=0.001, slippage_bps=5.0)
    executor = CCXTExecutor(
        exchange_id="binance",
        api_key=api_key,
        api_secret=api_secret,
        paper_mode=True,
        cost_model=cost_model,
    )

    # Fetch historical data for warmup
    print("Fetching historical data for warmup...")
    store = LiveDataStore(lookback_days=strategy.warmup_periods() + 30)

    for symbol in universe:
        print(f"  {symbol}...", end=" ")
        df = executor.fetch_historical(symbol, days=strategy.warmup_periods() + 30)
        if len(df) > 0:
            store.add_historical(symbol, df)
            print(f"OK ({len(df)} bars)")
        else:
            print("FAILED")

    # Create engine
    engine = Engine(
        strategy=strategy,
        data_store=store,
        mode=EngineMode.PAPER,
        executor=executor,
        initial_capital=initial_capital,
    )

    print("\nStarting paper trading...")
    print("Press Ctrl+C to stop\n")

    iteration = 0

    try:
        while True:
            now = datetime.now()
            prices = executor.fetch_prices(universe)

            if not prices:
                print(f"[{now.strftime('%H:%M:%S')}] No prices")
                time.sleep(60)
                continue

            result = engine.step(now, prices)

            if result.action == "rebalance":
                print(f"\n[{now.strftime('%Y-%m-%d %H:%M:%S')}] REBALANCE")
                print(f"  Fills: {len(result.fills)}")
                for fill in result.fills:
                    side = "BUY" if fill.side.value == "buy" else "SELL"
                    print(f"    {side} {fill.symbol}: {fill.amount:.4f} @ ${fill.price:,.2f}")
                print(f"  Equity: ${result.snapshot.equity:,.2f}")
                print(f"  Cash: ${result.snapshot.cash:,.2f}")
                print(f"  Positions: {result.snapshot.num_positions}")
            elif iteration % 10 == 0:
                print(
                    f"[{now.strftime('%H:%M:%S')}] Skip ({result.rebalance_reason}) | "
                    f"Equity: ${result.snapshot.equity:,.2f}"
                )

            iteration += 1
            time.sleep(60)

    except KeyboardInterrupt:
        print("\n\nStopping paper trading...")

        if len(engine.trades) > 0:
            print(f"\nTotal trades: {len(engine.trades)}")
            final_equity = engine.snapshots[-1].equity if engine.snapshots else initial_capital
            pnl = final_equity - initial_capital
            pnl_pct = (pnl / initial_capital) * 100
            print(f"Final equity: ${final_equity:,.2f}")
            print(f"P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)")
        else:
            print("\nNo trades executed")

    finally:
        executor.close()
        print("Done")


if __name__ == "__main__":
    main()
