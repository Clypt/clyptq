"""
Unit tests for Engine Core - Critical Test 4: Rebalancing Frequency

Tests that rebalancing occurs only once per period.
"""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from clyptq import Constraints, CostModel, EngineMode
from clyptq.data.store import DataStore
from clyptq.engine import Engine
from clyptq.execution import BacktestExecutor
from clyptq.core.base import Factor
from clyptq.portfolio.construction import TopNConstructor
from clyptq.strategy.base import SimpleStrategy
from typing import Dict


class ConstantFactor(Factor):
    """Simple factor that returns constant scores for testing."""

    def __init__(self, scores: Dict[str, float]):
        super().__init__(name="Constant")
        self._scores = scores

    def compute(self, data) -> Dict[str, float]:
        # Filter scores to only available symbols
        return {s: score for s, score in self._scores.items() if s in data.symbols}


def create_test_datastore():
    """Create a test datastore with daily data."""
    store = DataStore()

    # Create 30 days of data for 3 symbols
    dates = pd.date_range(start=datetime(2023, 1, 1), periods=30, freq="D")

    for symbol in ["BTC/USDT", "ETH/USDT", "BNB/USDT"]:
        data = pd.DataFrame({
            "open": [100.0 + i for i in range(30)],
            "high": [101.0 + i for i in range(30)],
            "low": [99.0 + i for i in range(30)],
            "close": [100.5 + i for i in range(30)],
            "volume": [1000.0] * 30,
        }, index=dates)

        store.add_ohlcv(symbol, data)

    return store


def test_daily_rebalance_once_per_day():
    """
    CRITICAL TEST 4: Rebalancing frequency control.

    engine/core.py:138-180 - MUST NOT rebalance twice on same day.

    Tests that with daily schedule, rebalancing happens exactly once per day,
    even if _process_timestamp is called multiple times on the same day.
    """
    # Create test data
    store = create_test_datastore()

    # Create simple strategy
    factor = ConstantFactor({"BTC/USDT": 1.0, "ETH/USDT": 0.8, "BNB/USDT": 0.6})
    constructor = TopNConstructor(top_n=2)
    constraints = Constraints(max_position_size=0.5, max_gross_exposure=1.0)

    strategy = SimpleStrategy(
        factors_list=[factor],
        constructor=constructor,
        constraints_obj=constraints,
        schedule_str="daily",
        warmup=0,
        name="TestStrategy",
    )

    # Create engine
    cost_model = CostModel(maker_fee=0.0, taker_fee=0.0, slippage_bps=0.0)
    executor = BacktestExecutor(cost_model)

    engine = Engine(
        strategy=strategy,
        data_store=store,
        executor=executor,
        initial_capital=10000.0,
    )

    # Process timestamp at 9:00 AM on Jan 1
    engine._process_timestamp(datetime(2023, 1, 1, 9, 0))
    trades_count_morning = len(engine.trades)

    # Process again at 2:00 PM on the SAME day
    engine._process_timestamp(datetime(2023, 1, 1, 14, 0))
    trades_count_afternoon = len(engine.trades)

    # Should NOT have created new trades (already rebalanced today)
    assert trades_count_afternoon == trades_count_morning, \
        "Should not rebalance twice on the same day"

    # Process next day - should rebalance
    engine._process_timestamp(datetime(2023, 1, 2, 9, 0))
    trades_count_next_day = len(engine.trades)

    # Should have new trades on next day
    assert trades_count_next_day > trades_count_afternoon, \
        "Should rebalance on next day"


def test_weekly_rebalance_once_per_week():
    """Test that weekly schedule rebalances only once per week."""
    store = create_test_datastore()

    factor = ConstantFactor({"BTC/USDT": 1.0, "ETH/USDT": 0.8})
    constructor = TopNConstructor(top_n=2)
    constraints = Constraints()

    strategy = SimpleStrategy(
        factors_list=[factor],
        constructor=constructor,
        constraints_obj=constraints,
        schedule_str="weekly",
        warmup=0,
    )

    cost_model = CostModel(maker_fee=0.0, taker_fee=0.0, slippage_bps=0.0)
    executor = BacktestExecutor(cost_model)

    engine = Engine(
        strategy=strategy,
        data_store=store,
        executor=executor,
        initial_capital=10000.0,
    )

    # Monday (week 1)
    engine._process_timestamp(datetime(2023, 1, 2))  # Jan 2 is Monday
    trades_monday = len(engine.trades)

    # Tuesday (same week)
    engine._process_timestamp(datetime(2023, 1, 3))
    trades_tuesday = len(engine.trades)

    # Should NOT rebalance on Tuesday (same week)
    assert trades_tuesday == trades_monday

    # Next Monday (week 2)
    engine._process_timestamp(datetime(2023, 1, 9))
    trades_next_monday = len(engine.trades)

    # Should rebalance on next Monday (different week)
    assert trades_next_monday > trades_tuesday


def test_monthly_rebalance_once_per_month():
    """Test that monthly schedule rebalances only once per month."""
    store = create_test_datastore()

    factor = ConstantFactor({"BTC/USDT": 1.0})
    constructor = TopNConstructor(top_n=1)
    constraints = Constraints()

    strategy = SimpleStrategy(
        factors_list=[factor],
        constructor=constructor,
        constraints_obj=constraints,
        schedule_str="monthly",
        warmup=0,
    )

    cost_model = CostModel(maker_fee=0.0, taker_fee=0.0, slippage_bps=0.0)
    executor = BacktestExecutor(cost_model)

    engine = Engine(
        strategy=strategy,
        data_store=store,
        executor=executor,
        initial_capital=10000.0,
    )

    # Jan 1
    engine._process_timestamp(datetime(2023, 1, 1))
    trades_jan1 = len(engine.trades)

    # Jan 15 (same month)
    engine._process_timestamp(datetime(2023, 1, 15))
    trades_jan15 = len(engine.trades)

    # Should NOT rebalance on Jan 15 (same month)
    assert trades_jan15 == trades_jan1

    # Jan 31 (still same month)
    engine._process_timestamp(datetime(2023, 1, 31))
    trades_jan31 = len(engine.trades)

    # Still should NOT rebalance
    assert trades_jan31 == trades_jan1


def test_should_rebalance_logic():
    """Test the _should_rebalance method directly."""
    store = create_test_datastore()

    factor = ConstantFactor({"BTC/USDT": 1.0})
    constructor = TopNConstructor(top_n=1)
    constraints = Constraints()

    # Daily schedule
    strategy_daily = SimpleStrategy(
        factors_list=[factor],
        constructor=constructor,
        constraints_obj=constraints,
        schedule_str="daily",
        warmup=0,
    )

    cost_model = CostModel()
    executor = BacktestExecutor(cost_model)

    engine = Engine(
        strategy=strategy_daily,
        data_store=store,
        executor=executor,
        initial_capital=10000.0,
    )

    # First call - should rebalance
    assert engine._should_rebalance(datetime(2023, 1, 1, 9, 0)) == True

    # Same day, different time - should NOT rebalance
    assert engine._should_rebalance(datetime(2023, 1, 1, 15, 0)) == False

    # Next day - should rebalance
    assert engine._should_rebalance(datetime(2023, 1, 2, 9, 0)) == True


def test_backtest_with_multiple_timestamps_per_day():
    """Test backtest with high-frequency data (multiple timestamps per day)."""
    store = DataStore()

    # Create hourly data
    dates = pd.date_range(
        start=datetime(2023, 1, 1),
        periods=48,  # 2 days of hourly data
        freq="h"
    )

    data = pd.DataFrame({
        "open": [100.0 + i * 0.1 for i in range(48)],
        "high": [101.0 + i * 0.1 for i in range(48)],
        "low": [99.0 + i * 0.1 for i in range(48)],
        "close": [100.5 + i * 0.1 for i in range(48)],
        "volume": [1000.0] * 48,
    }, index=dates)

    store.add_ohlcv("BTC/USDT", data)

    # Create strategy with daily rebalancing
    factor = ConstantFactor({"BTC/USDT": 1.0})
    constructor = TopNConstructor(top_n=1)
    constraints = Constraints()

    strategy = SimpleStrategy(
        factors_list=[factor],
        constructor=constructor,
        constraints_obj=constraints,
        schedule_str="daily",
        warmup=0,
    )

    cost_model = CostModel(maker_fee=0.0, taker_fee=0.0, slippage_bps=0.0)
    executor = BacktestExecutor(cost_model)

    engine = Engine(
        strategy=strategy,
        data_store=store,
        executor=executor,
        initial_capital=10000.0,
    )

    # Run backtest (will use daily schedule despite hourly data)
    result = engine.run(
        start=datetime(2023, 1, 1),
        end=datetime(2023, 1, 2),
        verbose=False,
    )

    # Should only rebalance once per day = 2 rebalances total
    # Each rebalance creates buy orders
    # Since we're starting fresh, we expect trades only on rebalance days
    assert len(result.trades) >= 2, "Should have trades from rebalancing"


def test_order_generation_sells_first():
    """Test that sell orders are generated before buy orders."""
    from clyptq.core.types import Order, OrderSide

    store = create_test_datastore()

    factor = ConstantFactor({"BTC/USDT": 1.0, "ETH/USDT": 0.5})
    constructor = TopNConstructor(top_n=1)  # Will only hold top 1
    constraints = Constraints()

    strategy = SimpleStrategy(
        factors_list=[factor],
        constructor=constructor,
        constraints_obj=constraints,
        schedule_str="daily",
        warmup=0,
    )

    cost_model = CostModel()
    executor = BacktestExecutor(cost_model)

    engine = Engine(
        strategy=strategy,
        data_store=store,
        executor=executor,
        initial_capital=10000.0,
    )

    # Create scenario where we need to sell one asset and buy another
    current_weights = {"ETH/USDT": 0.5}
    target_weights = {"BTC/USDT": 0.5}
    equity = 10000.0
    prices = {"BTC/USDT": 100.0, "ETH/USDT": 100.0}

    orders = engine._generate_orders(current_weights, target_weights, equity, prices)

    # Find sell and buy orders
    sells = [o for o in orders if o.side == OrderSide.SELL]
    buys = [o for o in orders if o.side == OrderSide.BUY]

    # Both should exist
    assert len(sells) > 0, "Should have sell orders"
    assert len(buys) > 0, "Should have buy orders"

    # Sells should come first (check order)
    sell_indices = [i for i, o in enumerate(orders) if o.side == OrderSide.SELL]
    buy_indices = [i for i, o in enumerate(orders) if o.side == OrderSide.BUY]

    assert max(sell_indices) < min(buy_indices), "Sell orders must come before buy orders"
