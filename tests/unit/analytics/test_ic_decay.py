import numpy as np
import pandas as pd
from datetime import datetime

from clyptq.analytics.factors.analyzer import FactorAnalyzer
from clyptq.analytics.factors.turnover import turnover_performance_frontier
from clyptq.data.stores.store import DataStore
from clyptq.trading.factors.library.momentum import MomentumFactor
from clyptq.core.types import BacktestResult, Snapshot, EngineMode


def create_test_store(n_days=50):
    store = DataStore()
    dates = pd.date_range(start=datetime(2024, 1, 1), periods=n_days, freq="D")

    np.random.seed(42)
    symbols = ["A", "B", "C", "D", "E"]

    for symbol in symbols:
        prices = 100 + np.cumsum(np.random.randn(n_days) * 2)
        data = pd.DataFrame(
            {
                "open": prices,
                "high": prices + 1,
                "low": prices - 1,
                "close": prices,
                "volume": [10000] * n_days,
            },
            index=dates,
        )
        store.add_ohlcv(symbol, data)

    return store


def test_ic_decay_basic():
    store = create_test_store(n_days=50)
    factor = MomentumFactor(lookback=10)
    analyzer = FactorAnalyzer()

    decay_df = analyzer.ic_decay_analysis(factor, store, max_horizon=10)

    assert not decay_df.empty
    assert "horizon" in decay_df.columns
    assert "mean_ic" in decay_df.columns
    assert "std_ic" in decay_df.columns
    assert "abs_mean_ic" in decay_df.columns
    assert len(decay_df) == 10


def test_ic_decay_horizons():
    store = create_test_store(n_days=50)
    factor = MomentumFactor(lookback=10)
    analyzer = FactorAnalyzer()

    decay_df = analyzer.ic_decay_analysis(factor, store, max_horizon=5)

    assert len(decay_df) == 5
    assert decay_df["horizon"].tolist() == [1, 2, 3, 4, 5]


def test_ic_decay_insufficient_data():
    store = create_test_store(n_days=10)
    factor = MomentumFactor(lookback=5)
    analyzer = FactorAnalyzer()

    decay_df = analyzer.ic_decay_analysis(factor, store, max_horizon=20)

    assert decay_df.empty


def test_turnover_performance_frontier_basic():
    snapshots = [
        Snapshot(
            timestamp=datetime(2024, 1, 1),
            cash=10000,
            equity=10000,
            positions={},
            positions_value=0,
            leverage=0,
            num_positions=0,
        ),
        Snapshot(
            timestamp=datetime(2024, 2, 1),
            cash=11000,
            equity=11000,
            positions={},
            positions_value=0,
            leverage=0,
            num_positions=0,
        ),
    ]

    result1 = BacktestResult(
        snapshots=snapshots,
        trades=[],
        metrics={"sharpe_ratio": 1.5, "total_return": 0.1, "max_drawdown": -0.05},
        strategy_name="test_strategy",
        mode=EngineMode.BACKTEST,
    )

    result2 = BacktestResult(
        snapshots=snapshots,
        trades=[],
        metrics={"sharpe_ratio": 1.2, "total_return": 0.08, "max_drawdown": -0.03},
        strategy_name="test_strategy",
        mode=EngineMode.BACKTEST,
    )

    df = turnover_performance_frontier([result1, result2], [0.5, 0.3])

    assert len(df) == 2
    assert "turnover" in df.columns
    assert "sharpe_ratio" in df.columns
    assert "net_sharpe" in df.columns


def test_turnover_performance_frontier_sorted():
    snapshots = [
        Snapshot(
            timestamp=datetime(2024, 1, 1),
            cash=10000,
            equity=10000,
            positions={},
            positions_value=0,
            leverage=0,
            num_positions=0,
        )
    ]

    results = [
        BacktestResult(
            snapshots=snapshots,
            trades=[],
            metrics={"sharpe_ratio": 1.5, "total_return": 0.1, "max_drawdown": -0.05},
            strategy_name="test_strategy",
            mode=EngineMode.BACKTEST,
        )
        for _ in range(3)
    ]

    df = turnover_performance_frontier(results, [0.8, 0.3, 0.5])

    assert df["turnover"].tolist() == [0.3, 0.5, 0.8]
