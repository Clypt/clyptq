"""
Paper Trading Example

Usage:
    python examples/paper_trading.py

Note: Press Ctrl+C to stop
"""

import os
import time
from datetime import datetime

from clyptq import CostModel
from clyptq.core.types import EngineMode
from clyptq.data.stores.live_store import LiveDataStore
from clyptq.trading.engine import LiveEngine
from clyptq.trading.execution.live import CCXTExecutor

from strategy import MomentumStrategy


def main():
    print("=" * 70)
    print("PAPER TRADING MODE - Live Prices, No Real Orders")
    print("=" * 70)

    # 1. Setup
    strategy = MomentumStrategy()
    universe = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT"]
    initial_capital = 10000.0

    print(f"\nStrategy: {strategy.name}")
    print(f"Universe: {', '.join(universe)}")
    print(f"Capital: ${initial_capital:,.0f}\n")

    # 2. Executor (paper mode)
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

    # 3. Warmup data
    print("Fetching warmup data...")
    store = LiveDataStore(lookback_days=strategy.warmup_periods() + 30)

    for symbol in universe:
        print(f"  {symbol}...", end=" ")
        df = executor.fetch_historical(symbol, days=strategy.warmup_periods() + 30)
        if len(df) > 0:
            store.add_historical(symbol, df)
            print(f"OK ({len(df)} bars)")
        else:
            print("FAILED")

    # 4. Engine
    engine = LiveEngine(
        strategy=strategy,
        data_store=store,
        executor=executor,
        initial_capital=initial_capital,
        mode=EngineMode.PAPER,
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
            elif iteration % 10 == 0:
                print(
                    f"[{now.strftime('%H:%M:%S')}] {result.rebalance_reason} | "
                    f"Equity: ${result.snapshot.equity:,.2f}"
                )

            iteration += 1
            time.sleep(60)

    except KeyboardInterrupt:
        print("\n\nStopping...")

        if len(engine.trades) > 0:
            final_equity = engine.snapshots[-1].equity
            pnl = final_equity - initial_capital
            pnl_pct = (pnl / initial_capital) * 100
            print(f"\nTrades: {len(engine.trades)}")
            print(f"Final equity: ${final_equity:,.2f}")
            print(f"P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)")
        else:
            print("\nNo trades")

    finally:
        executor.close()


if __name__ == "__main__":
    main()
