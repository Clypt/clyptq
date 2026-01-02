"""
Live Trading Example

WARNING: This uses REAL MONEY. Test thoroughly in paper mode first.

Usage:
    export BINANCE_API_KEY=your_key
    export BINANCE_API_SECRET=your_secret
    python examples/live_trading.py

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
    print("LIVE TRADING MODE - REAL MONEY AT RISK")
    print("=" * 70)

    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")

    if not api_key or not api_secret:
        print("\nERROR: Set BINANCE_API_KEY and BINANCE_API_SECRET environment variables")
        return

    # Confirmation
    print("\nWARNING: This will execute REAL trades with REAL money")
    confirm = input("Type 'YES' to continue: ")
    if confirm != "YES":
        print("Cancelled")
        return

    # 1. Setup
    strategy = MomentumStrategy()
    universe = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
    initial_capital = 1000.0

    print(f"\nStrategy: {strategy.name}")
    print(f"Universe: {', '.join(universe)}")
    print(f"Capital: ${initial_capital:,.0f}\n")

    # 2. Executor (live mode)
    cost_model = CostModel(maker_fee=0.001, taker_fee=0.001, slippage_bps=5.0)
    executor = CCXTExecutor(
        exchange_id="binance",
        api_key=api_key,
        api_secret=api_secret,
        paper_mode=False,
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
        mode=EngineMode.LIVE,
    )

    print("\nStarting LIVE trading...")
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
                print(f"\n[{now.strftime('%Y-%m-%d %H:%M:%S')}] REBALANCE (LIVE)")
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
