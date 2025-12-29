"""Paper trading command."""

import os
import time
from datetime import datetime

from clyptq import EngineMode
from clyptq.cli.commands.backtest import load_strategy_from_file
from clyptq.data.store import DataStore
from clyptq.engine.core import Engine
from clyptq.execution.live import CCXTExecutor
from clyptq.risk import CostModel


def handle_paper(args):
    """Run paper trading with live prices, no real orders."""
    print(f"\n{'='*70}")
    print("PAPER TRADING MODE")
    print(f"{'='*70}\n")

    # get API credentials
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")

    if not api_key or not api_secret:
        print("Error: Set BINANCE_API_KEY and BINANCE_API_SECRET")
        return

    # load strategy
    print(f"Loading strategy: {args.strategy}")
    StrategyClass = load_strategy_from_file(args.strategy)
    strategy = StrategyClass()
    print(f"Strategy: {strategy.name}")
    print(f"Capital: ${args.capital:,.0f}\n")

    # setup executor (paper mode)
    cost_model = CostModel(maker_fee=0.001, taker_fee=0.001, slippage_bps=5.0)
    executor = CCXTExecutor(
        exchange_id="binance",
        api_key=api_key,
        api_secret=api_secret,
        paper_mode=True,
        cost_model=cost_model,
    )

    # create minimal data store (live trading doesn't use historical data)
    store = DataStore()

    # create engine
    engine = Engine(
        strategy=strategy,
        data_store=store,
        mode=EngineMode.PAPER,
        executor=executor,
        initial_capital=args.capital,
    )

    print("Starting paper trading...")
    print("Press Ctrl+C to stop\n")

    try:
        # simple polling loop
        while True:
            now = datetime.now()
            print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Checking prices...")

            # fetch current prices
            symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
            prices = executor.fetch_prices(symbols)

            for sym, price in prices.items():
                print(f"  {sym}: ${price:,.2f}")

            # portfolio state
            equity = engine.portfolio.equity
            cash = engine.portfolio.cash
            print(f"  Equity: ${equity:,.2f} | Cash: ${cash:,.2f}")

            # wait before next check
            time.sleep(60)

    except KeyboardInterrupt:
        print("\n\nStopping paper trading...")

    finally:
        executor.close()
        print("Done")
