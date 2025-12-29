"""
Tests for walk-forward optimization.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from clyptq.optimization.walk_forward import WalkForwardOptimizer
from clyptq.strategy.base import SimpleStrategy
from clyptq.factors.library.momentum import MomentumFactor
from clyptq.portfolio.construction import TopNConstructor
from clyptq.core.types import Constraints
from clyptq.data.store import DataStore


def create_test_datastore(symbols, days=365):
    """Helper to create test data."""
    start = datetime(2024, 1, 1)
    store = DataStore()

    for symbol in symbols:
        dates = pd.date_range(start, periods=days, freq="D")
        prices = 100 + np.cumsum(np.random.randn(days) * 2)
        df = pd.DataFrame(
            {
                "open": prices,
                "high": prices * 1.01,
                "low": prices * 0.99,
                "close": prices,
                "volume": np.random.rand(days) * 1000,
            },
            index=dates,
        )
        store.add_ohlcv(symbol, df)

    return store


def momentum_strategy_factory(lookback=20, top_n=5):
    """Strategy factory for testing."""
    return SimpleStrategy(
        factors_list=[MomentumFactor(lookback=lookback)],
        constructor=TopNConstructor(top_n=top_n),
        constraints_obj=Constraints(
            max_position_size=0.3,
            min_position_size=0.05,
            max_gross_exposure=1.0,
            max_num_positions=10,
        ),
        warmup=25,
    )


def test_walk_forward_optimizer_init():
    """Test WalkForwardOptimizer initialization."""
    optimizer = WalkForwardOptimizer(
        strategy_factory=momentum_strategy_factory,
        param_grid={"lookback": [10, 20], "top_n": [3, 5]},
        train_days=180,
        test_days=30,
        metric="sharpe_ratio",
    )

    assert optimizer.train_days == 180
    assert optimizer.test_days == 30
    assert optimizer.metric == "sharpe_ratio"
    assert len(optimizer.param_grid) == 2


def test_generate_windows():
    """Test window generation."""
    optimizer = WalkForwardOptimizer(
        strategy_factory=momentum_strategy_factory,
        param_grid={"lookback": [20]},
        train_days=180,
        test_days=30,
    )

    start = datetime(2024, 1, 1)
    end = datetime(2024, 12, 31)

    windows = optimizer._generate_windows(start, end)

    assert len(windows) > 0
    for train_start, train_end, test_start, test_end in windows:
        assert train_end == test_start
        assert (train_end - train_start).days == 180
        assert (test_end - test_start).days == 30
        assert test_end <= end


def test_generate_param_combinations():
    """Test parameter combination generation."""
    optimizer = WalkForwardOptimizer(
        strategy_factory=momentum_strategy_factory,
        param_grid={"lookback": [10, 20, 30], "top_n": [3, 5]},
    )

    combinations = optimizer._generate_param_combinations()

    assert len(combinations) == 6
    assert {"lookback": 10, "top_n": 3} in combinations
    assert {"lookback": 30, "top_n": 5} in combinations


def test_extract_metric():
    """Test metric extraction."""
    from clyptq.core.types import BacktestResult, EngineMode, PerformanceMetrics

    metrics = PerformanceMetrics(
        total_return=0.25,
        annualized_return=0.25,
        daily_returns=[],
        volatility=0.1,
        sharpe_ratio=1.5,
        sortino_ratio=1.8,
        max_drawdown=-0.15,
        num_trades=10,
        win_rate=0.5,
        profit_factor=1.5,
        avg_trade_pnl=50.0,
        avg_leverage=1.0,
        max_leverage=1.5,
        avg_num_positions=5.0,
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 31),
        duration_days=365,
    )

    result = BacktestResult(
        snapshots=[],
        trades=[],
        metrics=metrics,
        strategy_name="Test",
        mode=EngineMode.BACKTEST,
    )

    optimizer_sharpe = WalkForwardOptimizer(
        strategy_factory=momentum_strategy_factory,
        param_grid={},
        metric="sharpe_ratio",
    )
    assert optimizer_sharpe._extract_metric(result) == 1.5

    optimizer_return = WalkForwardOptimizer(
        strategy_factory=momentum_strategy_factory,
        param_grid={},
        metric="total_return",
    )
    assert optimizer_return._extract_metric(result) == 0.25

    optimizer_dd = WalkForwardOptimizer(
        strategy_factory=momentum_strategy_factory,
        param_grid={},
        metric="max_drawdown",
    )
    assert optimizer_dd._extract_metric(result) == -0.15


def test_is_higher_better():
    """Test metric direction check."""
    optimizer_sharpe = WalkForwardOptimizer(
        strategy_factory=momentum_strategy_factory,
        param_grid={},
        metric="sharpe_ratio",
    )
    assert optimizer_sharpe._is_higher_better() is True

    optimizer_dd = WalkForwardOptimizer(
        strategy_factory=momentum_strategy_factory,
        param_grid={},
        metric="max_drawdown",
    )
    assert optimizer_dd._is_higher_better() is False


def test_walk_forward_optimize_small():
    """Test full walk-forward optimization on small dataset."""
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    store = create_test_datastore(symbols, days=300)

    optimizer = WalkForwardOptimizer(
        strategy_factory=momentum_strategy_factory,
        param_grid={"lookback": [10, 20], "top_n": [3, 5]},
        train_days=120,
        test_days=30,
        metric="sharpe_ratio",
        initial_capital=10000.0,
    )

    start = datetime(2024, 1, 1)
    end = datetime(2024, 6, 30)

    result = optimizer.optimize(store, start, end, verbose=False)

    assert len(result.windows) > 0
    assert result.avg_train_metric is not None
    assert result.avg_test_metric is not None
    assert len(result.best_params_frequency) > 0

    for window in result.windows:
        assert window.best_params is not None
        assert window.train_metric is not None
        assert window.test_metric is not None
        assert window.test_result is not None


def test_walk_forward_with_verbose():
    """Test walk-forward with verbose output."""
    symbols = ["BTC/USDT", "ETH/USDT"]
    store = create_test_datastore(symbols, days=250)

    optimizer = WalkForwardOptimizer(
        strategy_factory=momentum_strategy_factory,
        param_grid={"lookback": [20], "top_n": [5]},
        train_days=100,
        test_days=30,
    )

    start = datetime(2024, 1, 1)
    end = datetime(2024, 5, 31)

    result = optimizer.optimize(store, start, end, verbose=True)
    assert result is not None


def test_walk_forward_combined_result():
    """Test combined test result aggregation."""
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    store = create_test_datastore(symbols, days=300)

    optimizer = WalkForwardOptimizer(
        strategy_factory=momentum_strategy_factory,
        param_grid={"lookback": [20], "top_n": [5]},
        train_days=120,
        test_days=30,
    )

    start = datetime(2024, 1, 1)
    end = datetime(2024, 6, 30)

    result = optimizer.optimize(store, start, end)

    assert result.combined_test_result is not None
    assert len(result.combined_test_result.snapshots) > 0
    assert result.combined_test_result.metrics is not None
