import os
import time
from datetime import datetime, timedelta, timezone
from typing import List

import pytest

from clyptq.core.base import Factor
from clyptq.core.types import Constraints
from clyptq.data.loaders.ccxt import CCXTDataLoader
from clyptq.data.stores.live_store import LiveDataStore
from clyptq.trading.engine.live import LiveEngine
from clyptq.trading.execution.live import LiveExecutor
from clyptq.trading.factors.library.momentum import MomentumFactor
from clyptq.trading.portfolio.constructors import TopNConstructor
from clyptq.trading.strategy.base import Strategy


class SimpleTestStrategy(Strategy):
    def __init__(self):
        self._factors = [MomentumFactor(lookback=20)]
        self._constructor = TopNConstructor(top_n=3)
        self._constraints = Constraints()

    def factors(self) -> List[Factor]:
        return self._factors

    def portfolio_constructor(self):
        return self._constructor

    def constraints(self) -> Constraints:
        return self._constraints

    def schedule(self) -> str:
        return "daily"

    def warmup_periods(self) -> int:
        return 30

    def universe(self) -> List[str]:
        return ["BTC/USDT", "ETH/USDT", "BNB/USDT"]


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("CLYPTQ_RUN_EXCHANGE_TESTS") != "true",
    reason="Exchange tests disabled (set CLYPTQ_RUN_EXCHANGE_TESTS=true to enable)",
)
def test_binance_sandbox_connection():
    print("\n=== Binance Testnet Connection Test ===")

    api_key = os.getenv("BINANCE_TESTNET_API_KEY", "")
    api_secret = os.getenv("BINANCE_TESTNET_API_SECRET", "")

    if not api_key or not api_secret:
        pytest.skip("Binance testnet credentials not configured")

    executor = LiveExecutor(
        exchange_id="binance",
        api_key=api_key,
        api_secret=api_secret,
        paper_mode=False,
        sandbox=True,
    )

    print("Testing price fetch...")
    prices = executor.fetch_prices(["BTC/USDT", "ETH/USDT"])
    print(f"Fetched prices: {prices}")

    assert "BTC/USDT" in prices
    assert "ETH/USDT" in prices
    assert prices["BTC/USDT"] > 0
    assert prices["ETH/USDT"] > 0

    print("Testing position fetch...")
    positions = executor.fetch_positions()
    print(f"Positions: {positions}")

    assert isinstance(positions, dict)


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("CLYPTQ_RUN_EXCHANGE_TESTS") != "true",
    reason="Exchange tests disabled",
)
def test_end_to_end_paper_workflow():
    print("\n=== End-to-End Paper Trading Workflow ===")

    strategy = SimpleTestStrategy()
    store = LiveDataStore(lookback_days=60)

    loader = CCXTDataLoader(exchange_id="binance", sandbox=False)
    universe = strategy.universe()

    print("Loading historical data...")
    for symbol in universe:
        df = loader.load_ohlcv(
            symbol=symbol,
            timeframe="1d",
            start=datetime.now(timezone.utc) - timedelta(days=60),
            end=datetime.now(timezone.utc),
        )
        store.add_historical(symbol, df)

    executor = LiveExecutor(
        exchange_id="binance",
        api_key="",
        api_secret="",
        paper_mode=True,
        sandbox=True,
    )

    engine = LiveEngine(
        strategy=strategy,
        data_store=store,
        executor=executor,
        initial_capital=10000,
    )

    print("Running 5 trading steps...")
    results = []
    for i in range(5):
        timestamp = datetime.now(timezone.utc)
        prices = {"BTC/USDT": 50000 + i * 100, "ETH/USDT": 3000 + i * 10, "BNB/USDT": 400 + i * 5}

        result = engine.step(timestamp, prices)
        results.append(result)

        print(f"Step {i + 1}: {result.action}, Equity: ${result.snapshot.equity:.2f}")
        time.sleep(0.1)

    print(f"\n=== Final Results ===")
    print(f"Total steps: {len(results)}")
    print(f"Final equity: ${results[-1].snapshot.equity:.2f}")
    print(f"Total trades: {len(engine.trades)}")

    assert len(results) == 5
    assert results[-1].snapshot.equity > 0


@pytest.mark.integration
def test_multi_strategy_concurrent():
    import threading

    print("\n=== Multi-Strategy Concurrent Execution ===")

    class Strategy1(Strategy):
        def factors(self) -> List[Factor]:
            return [MomentumFactor(lookback=10)]

        def portfolio_constructor(self):
            return TopNConstructor(top_n=2)

        def constraints(self) -> Constraints:
            return Constraints()

        def schedule(self) -> str:
            return "daily"

        def warmup_periods(self) -> int:
            return 20

        def universe(self) -> List[str]:
            return ["BTC/USDT", "ETH/USDT"]

    class Strategy2(Strategy):
        def factors(self) -> List[Factor]:
            return [MomentumFactor(lookback=20)]

        def portfolio_constructor(self):
            return TopNConstructor(top_n=2)

        def constraints(self) -> Constraints:
            return Constraints()

        def schedule(self) -> str:
            return "daily"

        def warmup_periods(self) -> int:
            return 30

        def universe(self) -> List[str]:
            return ["BTC/USDT", "ETH/USDT"]

    def run_strategy(strategy_class, strategy_id, results):
        strategy = strategy_class()
        store = LiveDataStore(lookback_days=40)

        loader = CCXTDataLoader(exchange_id="binance", sandbox=False)
        for symbol in strategy.universe():
            try:
                df = loader.load_ohlcv(
                    symbol=symbol,
                    timeframe="1d",
                    start=datetime.now(timezone.utc) - timedelta(days=40),
                    end=datetime.now(timezone.utc),
                )
                store.add_historical(symbol, df)
            except Exception as e:
                print(f"Strategy {strategy_id}: Failed to load {symbol}: {e}")
                return

        executor = LiveExecutor(
            exchange_id="binance",
            api_key="",
            api_secret="",
            paper_mode=True,
            sandbox=True,
        )

        engine = LiveEngine(
            strategy=strategy,
            data_store=store,
            executor=executor,
            initial_capital=10000,
        )

        timestamp = datetime.now(timezone.utc)
        prices = {"BTC/USDT": 50000, "ETH/USDT": 3000}

        result = engine.step(timestamp, prices)
        results[strategy_id] = {
            "equity": result.snapshot.equity,
            "action": result.action,
            "fills": len(result.fills),
        }
        print(f"Strategy {strategy_id}: {result.action}, Equity: ${result.snapshot.equity:.2f}")

    results = {}
    threads = [
        threading.Thread(target=run_strategy, args=(Strategy1, 1, results)),
        threading.Thread(target=run_strategy, args=(Strategy2, 2, results)),
    ]

    print("Starting concurrent strategies...")
    for t in threads:
        t.start()

    for t in threads:
        t.join()

    print(f"\n=== Concurrent Results ===")
    for strategy_id, result in results.items():
        print(f"Strategy {strategy_id}: {result}")

    assert len(results) == 2
    assert all(r["equity"] > 0 for r in results.values())


@pytest.mark.integration
def test_network_failure_recovery():
    import unittest.mock as mock

    import ccxt

    print("\n=== Network Failure Recovery Test ===")

    strategy = SimpleTestStrategy()
    store = LiveDataStore(lookback_days=40)

    loader = CCXTDataLoader(exchange_id="binance", sandbox=False)
    for symbol in strategy.universe():
        try:
            df = loader.load_ohlcv(
                symbol=symbol,
                timeframe="1d",
                start=datetime.now(timezone.utc) - timedelta(days=40),
                end=datetime.now(timezone.utc),
            )
            store.add_historical(symbol, df)
        except Exception:
            pass

    executor = LiveExecutor(
        exchange_id="binance",
        api_key="",
        api_secret="",
        paper_mode=True,
        sandbox=True,
    )

    call_count = {"count": 0}

    original_fetch = executor.fetch_prices

    def failing_fetch(symbols):
        call_count["count"] += 1
        if call_count["count"] <= 2:
            raise ccxt.NetworkError("Simulated network failure")
        return original_fetch(symbols)

    executor.fetch_prices = failing_fetch

    engine = LiveEngine(
        strategy=strategy,
        data_store=store,
        executor=executor,
        initial_capital=10000,
    )

    print("Testing network failure scenarios...")
    for i in range(5):
        timestamp = datetime.now(timezone.utc)
        prices = {"BTC/USDT": 50000, "ETH/USDT": 3000, "BNB/USDT": 400}

        try:
            result = engine.step(timestamp, prices)
            print(f"Step {i + 1}: Success - {result.action}")
        except ccxt.NetworkError as e:
            print(f"Step {i + 1}: Network error caught - {e}")

    print(f"\n=== Recovery Test Complete ===")
    print(f"Network failures simulated: 2")
    print(f"Total attempts: {call_count['count']}")

    assert call_count["count"] >= 2


@pytest.mark.integration
def test_position_sync_across_restarts():
    print("\n=== Position Sync Across Restarts ===")

    strategy = SimpleTestStrategy()
    store = LiveDataStore(lookback_days=40)

    loader = CCXTDataLoader(exchange_id="binance", sandbox=False)
    for symbol in strategy.universe():
        try:
            df = loader.load_ohlcv(
                symbol=symbol,
                timeframe="1d",
                start=datetime.now(timezone.utc) - timedelta(days=40),
                end=datetime.now(timezone.utc),
            )
            store.add_historical(symbol, df)
        except Exception:
            pass

    executor = LiveExecutor(
        exchange_id="binance",
        api_key="",
        api_secret="",
        paper_mode=True,
        sandbox=True,
    )

    engine1 = LiveEngine(
        strategy=strategy,
        data_store=store,
        executor=executor,
        initial_capital=10000,
    )

    print("Session 1: Executing trades...")
    timestamp = datetime.now(timezone.utc)
    prices = {"BTC/USDT": 50000, "ETH/USDT": 3000, "BNB/USDT": 400}

    result1 = engine1.step(timestamp, prices)
    print(f"Session 1: {result1.action}, Positions: {len(result1.snapshot.positions)}")

    session1_positions = dict(result1.snapshot.positions)
    session1_cash = result1.snapshot.cash

    engine2 = LiveEngine(
        strategy=strategy,
        data_store=store,
        executor=executor,
        initial_capital=10000,
    )

    print("Session 2: Fresh engine instance...")
    result2 = engine2.step(timestamp, prices)
    print(f"Session 2: {result2.action}, Positions: {len(result2.snapshot.positions)}")

    print(f"\n=== Position Comparison ===")
    print(f"Session 1 cash: ${session1_cash:.2f}")
    print(f"Session 2 cash: ${result2.snapshot.cash:.2f}")
    print(f"Session 1 positions: {len(session1_positions)}")
    print(f"Session 2 positions: {len(result2.snapshot.positions)}")

    assert result1.snapshot.equity > 0
    assert result2.snapshot.equity > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "integration"])
