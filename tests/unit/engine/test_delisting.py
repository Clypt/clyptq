"""Test delisting detection and forced liquidation."""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from clyptq.data.stores.store import DataStore
from clyptq.engine import Engine
from clyptq.execution.backtest import BacktestExecutor
from clyptq.factors.library.momentum import MomentumFactor
from clyptq.portfolio.constructors import TopNConstructor
from clyptq.portfolio.constraints import Constraints
from clyptq.strategy.base import SimpleStrategy
from clyptq.core.types import CostModel, EngineMode


def test_delisting_forces_liquidation():
    """Symbol gets delisted mid-backtest, position liquidated immediately."""
    store = DataStore()

    # BTC listed for full period
    start = datetime(2024, 1, 1)
    end = datetime(2024, 2, 1)
    dates = pd.date_range(start, end, freq="D")

    btc_data = pd.DataFrame(
        {
            "open": 40000.0,
            "high": 41000.0,
            "low": 39000.0,
            "close": 40000.0,
            "volume": 1000.0,
        },
        index=dates,
    )
    store.add_ohlcv("BTC/USDT", btc_data)

    # ETH delists on Jan 15
    delist_date = datetime(2024, 1, 15)
    eth_dates = pd.date_range(start, delist_date, freq="D")

    eth_data = pd.DataFrame(
        {
            "open": 2500.0,
            "high": 2600.0,
            "low": 2400.0,
            "close": 2500.0,
            "volume": 500.0,
        },
        index=eth_dates,
    )
    store.add_ohlcv("ETH/USDT", eth_data)

    strategy = SimpleStrategy(
        factors_list=[MomentumFactor(lookback=10)],
        constructor=TopNConstructor(top_n=2),
        constraints_obj=Constraints(),
        warmup=20,
    )
    cost_model = CostModel(maker_fee=0.001, taker_fee=0.001, slippage_bps=5)
    executor = BacktestExecutor(cost_model)
    engine = Engine(
        strategy=strategy,
        data_store=store,
        executor=executor,
        initial_capital=10000.0,
    )

    result = engine.run(start, end)

    # Check that ETH was sold when delisted
    eth_sells = [
        t for t in result.trades
        if t.symbol == "ETH/USDT" and t.side.value == "sell"
    ]

    assert len(eth_sells) > 0, "ETH should have been sold"

    # Check no ETH position after delisting (or amount=0)
    post_delist_snapshots = [
        s for s in result.snapshots
        if s.timestamp > delist_date + timedelta(days=1)
    ]

    for snapshot in post_delist_snapshots:
        if "ETH/USDT" in snapshot.positions:
            assert snapshot.positions["ETH/USDT"].amount == 0.0, "ETH should be liquidated"


def test_no_liquidation_if_not_delisted():
    """Symbols stay listed, no forced liquidation."""
    store = DataStore()

    start = datetime(2024, 1, 1)
    end = datetime(2024, 2, 1)
    dates = pd.date_range(start, end, freq="D")

    for symbol in ["BTC/USDT", "ETH/USDT"]:
        df = pd.DataFrame(
            {
                "open": 1000.0,
                "high": 1100.0,
                "low": 900.0,
                "close": 1000.0,
                "volume": 100.0,
            },
            index=dates,
        )
        store.add_ohlcv(symbol, df)

    strategy = SimpleStrategy(
        factors_list=[MomentumFactor(lookback=10)],
        constructor=TopNConstructor(top_n=2),
        constraints_obj=Constraints(),
        warmup=20,
    )
    cost_model = CostModel(maker_fee=0.001, taker_fee=0.001, slippage_bps=5)
    executor = BacktestExecutor(cost_model)
    engine = Engine(
        strategy=strategy,
        data_store=store,
        executor=executor,
        initial_capital=10000.0,
    )

    result = engine.run(start, end)

    # No forced liquidations, only regular rebalancing
    assert len(result.trades) > 0
