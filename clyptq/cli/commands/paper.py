"""Paper trading command."""

import os
import time
from datetime import datetime

from clyptq import EngineMode
from clyptq.cli.commands.backtest import load_strategy_from_file
from clyptq.data.live_store import LiveDataStore
from clyptq.engine import Engine
from clyptq.execution.live import CCXTExecutor
from clyptq.risk import CostModel


def handle_paper(args):
    """Run paper trading with live prices, no real orders."""
    print(f"\n{'='*70}")
    print("PAPER TRADING MODE")
    print(f"{'='*70}\n")

    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")

    print(f"Loading strategy: {args.strategy}")
    StrategyClass = load_strategy_from_file(args.strategy)
    strategy = StrategyClass()
    print(f"Strategy: {strategy.name}")
    print(f"Capital: ${args.capital:,.0f}")

    universe = strategy.universe()
    if not universe:
        print("Error: Strategy must define universe()")
        return

    print(f"Universe: {', '.join(universe)}")
    print(f"Schedule: {strategy.schedule()}")
    print(f"Warmup: {strategy.warmup_periods()} periods\n")

    cost_model = CostModel(maker_fee=0.001, taker_fee=0.001, slippage_bps=5.0)
    executor = CCXTExecutor(
        exchange_id="binance",
        api_key=api_key,
        api_secret=api_secret,
        paper_mode=True,
        cost_model=cost_model,
    )

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

    engine = Engine(
        strategy=strategy,
        data_store=store,
        mode=EngineMode.PAPER,
        executor=executor,
        initial_capital=args.capital,
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
                print(f"[{now.strftime('%H:%M:%S')}] Skip ({result.rebalance_reason}) | Equity: ${result.snapshot.equity:,.2f}")

            iteration += 1
            time.sleep(60)

    except KeyboardInterrupt:
        print("\n\nStopping paper trading...")

        if len(engine.trades) > 0:
            print(f"\nTotal trades: {len(engine.trades)}")
            final_equity = engine.snapshots[-1].equity if engine.snapshots else args.capital
            pnl = final_equity - args.capital
            pnl_pct = (pnl / args.capital) * 100
            print(f"Final equity: ${final_equity:,.2f}")
            print(f"P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)")

    finally:
        executor.close()
        print("Done")
