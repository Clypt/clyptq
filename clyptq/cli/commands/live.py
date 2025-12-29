"""Live trading command."""

import time
from datetime import datetime

from clyptq import EngineMode
from clyptq.cli.commands.backtest import load_strategy_from_file
from clyptq.data.store import DataStore
from clyptq.engine.core import Engine
from clyptq.execution.live import CCXTExecutor
from clyptq.risk import CostModel


def handle_live(args):
    """Run live trading with real money."""
    print(f"\n{'='*70}")
    print("LIVE TRADING MODE - REAL MONEY")
    print(f"{'='*70}\n")

    # safety check
    if not args.api_key or not args.api_secret:
        print("Error: --api-key and --api-secret required")
        return

    # confirmation
    print("WARNING: This will trade with REAL MONEY")
    response = input("Type 'YES' to continue: ")
    if response != "YES":
        print("Aborted")
        return

    # load strategy
    print(f"\nLoading strategy: {args.strategy}")
    StrategyClass = load_strategy_from_file(args.strategy)
    strategy = StrategyClass()
    print(f"Strategy: {strategy.name}\n")

    # setup executor (live mode)
    cost_model = CostModel(maker_fee=0.001, taker_fee=0.001, slippage_bps=5.0)
    executor = CCXTExecutor(
        exchange_id=args.exchange,
        api_key=args.api_key,
        api_secret=args.api_secret,
        paper_mode=False,  # real money
        sandbox=False,     # production
        cost_model=cost_model,
    )

    # create minimal data store
    store = DataStore()

    # get initial balance
    balance = executor.get_balance("USDT")
    print(f"USDT Balance: ${balance:,.2f}")

    if balance < 100:
        print("Error: Insufficient balance (min $100)")
        executor.close()
        return

    # create engine
    engine = Engine(
        strategy=strategy,
        data_store=store,
        mode=EngineMode.LIVE,
        executor=executor,
        initial_capital=balance,
    )

    print("\nStarting live trading...")
    print("Press Ctrl+C to stop\n")

    try:
        # polling loop
        while True:
            now = datetime.now()
            print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Live check...")

            # fetch prices
            symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
            prices = executor.fetch_prices(symbols)

            for sym, price in prices.items():
                print(f"  {sym}: ${price:,.2f}")

            # fetch actual positions
            positions = executor.fetch_positions()
            if positions:
                print(f"  Positions: {list(positions.keys())}")

            # portfolio state
            snapshot = engine.portfolio.get_snapshot(now, prices)
            print(f"  Equity: ${snapshot.equity:,.2f} | Cash: ${snapshot.cash:,.2f}")

            # wait
            time.sleep(60)

    except KeyboardInterrupt:
        print("\n\nStopping live trading...")

    finally:
        executor.close()
        print("Done")
