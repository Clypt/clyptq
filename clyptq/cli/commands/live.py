"""Live trading command.

Uses LiveEngine for real trading:
- CCXTCollector: Data fetching (historical + live)
- CCXTExecutor: Order execution
- LiveSource: Live data streaming with buffer
- LiveEngine: Trading loop
"""

from datetime import datetime, timedelta

from clyptq.cli.commands.backtest import load_strategy_from_file
from clyptq.data.collectors.ccxt import CCXTCollector
from clyptq.data.sources.live import LiveSource
from clyptq.trading.engine.live import LiveEngine
from clyptq.trading.execution import CCXTExecutor
from clyptq.core.types import CostModel


def handle_live(args):
    """Run live trading with real money.

    Architecture:
    - CCXTCollector: Fetches historical data for warmup + subscribes to live data
    - CCXTExecutor: Executes real orders
    - LiveSource: Live data with internal buffer
    - LiveEngine: Trading loop
    """
    print(f"\n{'='*70}")
    print("LIVE TRADING MODE - REAL MONEY")
    print(f"{'='*70}\n")

    if not args.api_key or not args.api_secret:
        print("Error: --api-key and --api-secret required")
        return

    print("WARNING: This will trade with REAL MONEY")
    response = input("Type 'YES' to continue: ")
    if response != "YES":
        print("Aborted")
        return

    print(f"\nLoading strategy: {args.strategy}")
    StrategyClass = load_strategy_from_file(args.strategy)
    strategy = StrategyClass()
    print(f"Strategy: {strategy.name}")

    universe = strategy.universe()
    if not universe:
        print("Error: Strategy must define universe()")
        return

    print(f"Universe: {', '.join(universe)}")
    print(f"Schedule: {strategy.schedule()}")
    print(f"Warmup: {strategy.warmup_periods()} periods\n")

    # Create executor (order execution only)
    cost_model = CostModel(maker_fee=0.001, taker_fee=0.001, slippage_bps=5.0)
    executor = CCXTExecutor(
        exchange_id=args.exchange,
        api_key=args.api_key,
        api_secret=args.api_secret,
        paper_mode=False,
        sandbox=False,
        cost_model=cost_model,
    )

    balance = executor.get_balance("USDT")
    print(f"USDT Balance: ${balance:,.2f}")

    if balance < 100:
        print("Error: Insufficient balance (min $100)")
        executor.close()
        return

    # Create data collector and live source
    print("\nInitializing data source...")
    collector = CCXTCollector(args.exchange)
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
        initial_capital=balance,
    )

    print("\nStarting live trading...")
    print("Press Ctrl+C to stop\n")

    try:
        engine.run_live(interval_seconds=60, verbose=True)

    except KeyboardInterrupt:
        print("\n\nStopping live trading...")

    finally:
        if len(engine.trades) > 0:
            print(f"\nTotal trades: {len(engine.trades)}")
            final_equity = engine.snapshots[-1].equity if engine.snapshots else balance
            pnl = final_equity - balance
            pnl_pct = (pnl / balance) * 100
            print(f"Final equity: ${final_equity:,.2f}")
            print(f"P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)")

        executor.close()
        collector.close()
        print("Done")
