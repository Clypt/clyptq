"""Paper trading command.

Uses LiveEngine in paper mode:
- CCXTExecutor with paper_mode=True: Simulates orders with live prices
- LiveSource: Live data streaming with buffer
- LiveEngine: Trading engine for live/paper trading
"""

import os
from datetime import datetime, timedelta

from clyptq.cli.commands.backtest import load_strategy_from_file
from clyptq.data.collectors.ccxt import CCXTCollector
from clyptq.data.sources.live import LiveSource
from clyptq.trading.engine.live import LiveEngine
from clyptq.trading.execution import CCXTExecutor
from clyptq.core.types import CostModel


def handle_paper(args):
    """Run paper trading with live prices, no real orders.

    Architecture:
    - CCXTCollector: Fetches historical data for warmup + subscribes to live data
    - CCXTExecutor: Paper mode - simulates orders with live prices
    - LiveSource: Live data with internal buffer
    - LiveEngine: Trading loop
    """
    print(f"\n{'='*70}")
    print("PAPER TRADING MODE")
    print(f"{'='*70}\n")

    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")
    exchange = getattr(args, "exchange", "binance")

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

    # Create executor in paper mode (simulates orders with live prices)
    cost_model = CostModel(maker_fee=0.001, taker_fee=0.001, slippage_bps=5.0)
    executor = CCXTExecutor(
        exchange_id=exchange,
        api_key=api_key,
        api_secret=api_secret,
        paper_mode=True,
        cost_model=cost_model,
    )

    # Create data collector and live source
    print("Initializing data source...")
    collector = CCXTCollector(exchange)
    source = LiveSource(collector=collector, timeframe="1d", buffer_size=500)

    # Load historical data for warmup
    warmup_days = strategy.warmup_periods() + 30
    print(f"\nFetching historical data for warmup ({warmup_days} days)...")

    start = datetime.now() - timedelta(days=warmup_days)
    end = datetime.now()

    data = source.load(
        symbols=universe,
        start=start,
        end=end,
        timeframe="1d",
    )

    loaded_count = len([s for s in universe if s in source.available_symbols()])
    print(f"Loaded {loaded_count}/{len(universe)} symbols")

    # Create LiveEngine
    engine = LiveEngine(
        strategy=strategy,
        source=source,
        executor=executor,
        initial_capital=args.capital,
    )

    print("\nStarting paper trading...")
    print("Press Ctrl+C to stop\n")

    try:
        engine.run_live(interval_seconds=60, verbose=True)

    except KeyboardInterrupt:
        print("\n\nStopping paper trading...")

    finally:
        if len(engine.trades) > 0:
            print(f"\nTotal trades: {len(engine.trades)}")
            final_equity = engine.snapshots[-1].equity if engine.snapshots else args.capital
            pnl = final_equity - args.capital
            pnl_pct = (pnl / args.capital) * 100
            print(f"Final equity: ${final_equity:,.2f}")
            print(f"P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)")

        executor.close()
        collector.close()
        print("Done")
