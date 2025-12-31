"""Tests for Engine.step() method."""

from datetime import datetime, timedelta

import pandas as pd

from clyptq import EngineMode
from clyptq.data.stores.live_store import LiveDataStore
from clyptq.trading.engine import LiveEngine
from clyptq.trading.execution.backtest import BacktestExecutor
from clyptq.trading.factors.library.momentum import MomentumFactor
from clyptq.trading.portfolio.constructors import TopNConstructor
from clyptq.trading.portfolio.constraints import Constraints
from clyptq.trading.risk.costs import CostModel
from clyptq.core.base import Strategy


class SimpleStrategy(Strategy):
    def __init__(self):
        super().__init__(name="TestStrategy")
        self._factors = [MomentumFactor(lookback=20)]
        self._constructor = TopNConstructor(top_n=3)
        self._constraints = Constraints()
        self._schedule = "daily"
        self._warmup = 25
        self._universe = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]

    def factors(self):
        return self._factors

    def portfolio_constructor(self):
        return self._constructor

    def constraints(self):
        return self._constraints

    def schedule(self):
        return self._schedule

    def warmup_periods(self):
        return self._warmup

    def universe(self):
        return self._universe


def test_step_skip_schedule():
    strategy = SimpleStrategy()
    store = LiveDataStore(lookback_days=60)

    df = pd.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(30)],
            "open": [100.0 for _ in range(30)],
            "high": [101.0 for _ in range(30)],
            "low": [99.0 for _ in range(30)],
            "close": [100.0 + i for i in range(30)],
            "volume": [1000.0 for _ in range(30)],
        }
    )

    for symbol in strategy.universe():
        store.add_historical(symbol, df)

    cost_model = CostModel()
    executor = BacktestExecutor(cost_model)
    engine = LiveEngine(strategy, store, executor, initial_capital=10000, mode=EngineMode.PAPER)

    prices = {"BTC/USDT": 130.0, "ETH/USDT": 130.0, "BNB/USDT": 130.0}

    result1 = engine.step(datetime(2024, 1, 31, 10, 0), prices)
    assert result1.action == "rebalance"

    result2 = engine.step(datetime(2024, 1, 31, 14, 0), prices)
    assert result2.action == "skip"
    assert result2.rebalance_reason == "schedule"


def test_step_rebalance():
    strategy = SimpleStrategy()
    store = LiveDataStore(lookback_days=60)

    df = pd.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(30)],
            "open": [100.0 for _ in range(30)],
            "high": [101.0 for _ in range(30)],
            "low": [99.0 for _ in range(30)],
            "close": [100.0 + i * 0.5 for i in range(30)],
            "volume": [1000.0 for _ in range(30)],
        }
    )

    for symbol in strategy.universe():
        store.add_historical(symbol, df)

    cost_model = CostModel()
    executor = BacktestExecutor(cost_model)
    engine = LiveEngine(strategy, store, executor, initial_capital=10000, mode=EngineMode.PAPER)

    prices = {"BTC/USDT": 115.0, "ETH/USDT": 115.0, "BNB/USDT": 115.0}

    result = engine.step(datetime(2024, 1, 31), prices)

    assert result.action == "rebalance"
    assert len(result.fills) > 0
    assert len(result.orders) > 0
    assert result.snapshot.equity > 0
    assert result.rebalance_reason == "scheduled"


def test_step_no_symbols():
    strategy = SimpleStrategy()
    store = LiveDataStore(lookback_days=60)

    cost_model = CostModel()
    executor = BacktestExecutor(cost_model)
    engine = LiveEngine(strategy, store, executor, initial_capital=10000, mode=EngineMode.PAPER)

    prices = {}

    result = engine.step(datetime(2024, 1, 31), prices)

    assert result.action == "skip"
    assert result.rebalance_reason == "no_symbols"
    assert len(result.fills) == 0


def test_step_updates_livestore():
    strategy = SimpleStrategy()
    store = LiveDataStore(lookback_days=60)

    df = pd.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(30)],
            "open": [100.0 for _ in range(30)],
            "high": [101.0 for _ in range(30)],
            "low": [99.0 for _ in range(30)],
            "close": [100.0 for _ in range(30)],
            "volume": [1000.0 for _ in range(30)],
        }
    )

    for symbol in strategy.universe():
        store.add_historical(symbol, df)

    cost_model = CostModel()
    executor = BacktestExecutor(cost_model)
    engine = LiveEngine(strategy, store, executor, initial_capital=10000, mode=EngineMode.PAPER)

    initial_len = len(store.data["BTC/USDT"])

    prices = {"BTC/USDT": 130.0, "ETH/USDT": 130.0, "BNB/USDT": 130.0}
    engine.step(datetime(2024, 2, 1), prices)

    assert len(store.data["BTC/USDT"]) == initial_len + 1
