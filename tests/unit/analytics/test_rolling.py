"""Tests for rolling metrics."""

from datetime import datetime, timedelta

import pytest

from clyptq.analytics.performance.rolling import RollingMetricsCalculator, RollingMetricsResult
from clyptq.core.types import (
    BacktestResult,
    Snapshot,
    PerformanceMetrics,
)


def create_test_result(days: int = 60) -> BacktestResult:
    """Create test backtest result with linear equity growth."""
    start = datetime(2024, 1, 1)
    snapshots = []

    for i in range(days):
        ts = start + timedelta(days=i)
        equity = 100000.0 + i * 100

        snap = Snapshot(
            timestamp=ts,
            equity=equity,
            cash=50000.0,
            positions={},
            positions_value=0.0,
        )
        snapshots.append(snap)

    metrics = PerformanceMetrics(
        total_return=0.06,
        annualized_return=0.12,
        volatility=0.15,
        sharpe_ratio=0.8,
        sortino_ratio=1.2,
        max_drawdown=0.05,
        num_trades=10,
        win_rate=0.6,
        profit_factor=1.5,
        avg_trade_pnl=100.0,
        avg_leverage=0.5,
        max_leverage=0.8,
        avg_num_positions=2.0,
        start_date=start,
        end_date=start + timedelta(days=days - 1),
        duration_days=days,
        daily_returns=[0.001] * (days - 1),
    )

    return BacktestResult(
        snapshots=snapshots,
        trades=[],
        metrics=metrics,
        strategy_name="Test",
        mode="backtest",
    )


def test_calculator_initialization():
    """Test calculator initialization."""
    calc = RollingMetricsCalculator(window=30, risk_free_rate=0.02)
    assert calc.window == 30
    assert calc.risk_free_rate == 0.02


def test_calculate_rolling_metrics():
    """Test basic rolling metrics calculation."""
    result = create_test_result(days=60)
    calc = RollingMetricsCalculator(window=30)

    rolling = calc.calculate(result)

    assert isinstance(rolling, RollingMetricsResult)
    assert len(rolling.timestamps) == 30
    assert len(rolling.sharpe_ratio) == 30
    assert len(rolling.sortino_ratio) == 30
    assert len(rolling.volatility) == 30
    assert len(rolling.max_drawdown) == 30
    assert len(rolling.returns) == 30


def test_rolling_sharpe_positive():
    """Test rolling Sharpe with positive returns."""
    result = create_test_result(days=60)
    calc = RollingMetricsCalculator(window=30)

    rolling = calc.calculate(result)

    for sharpe in rolling.sharpe_ratio:
        assert sharpe > 0


def test_rolling_volatility():
    """Test rolling volatility calculation."""
    result = create_test_result(days=60)
    calc = RollingMetricsCalculator(window=30)

    rolling = calc.calculate(result)

    for vol in rolling.volatility:
        assert vol >= 0


def test_rolling_max_drawdown():
    """Test rolling max drawdown calculation."""
    result = create_test_result(days=60)
    calc = RollingMetricsCalculator(window=30)

    rolling = calc.calculate(result)

    for dd in rolling.max_drawdown:
        assert dd <= 0


def test_insufficient_data_error():
    """Test error when insufficient data."""
    result = create_test_result(days=20)
    calc = RollingMetricsCalculator(window=30)

    with pytest.raises(ValueError, match="Need at least 30 snapshots"):
        calc.calculate(result)


def test_to_dict():
    """Test result serialization."""
    result = create_test_result(days=60)
    calc = RollingMetricsCalculator(window=30)

    rolling = calc.calculate(result)
    data = rolling.to_dict()

    assert "timestamps" in data
    assert "sharpe_ratio" in data
    assert "sortino_ratio" in data
    assert "volatility" in data
    assert "max_drawdown" in data
    assert "returns" in data

    assert len(data["timestamps"]) == 30
    assert isinstance(data["timestamps"][0], str)


def test_custom_window():
    """Test with custom window size."""
    result = create_test_result(days=100)
    calc = RollingMetricsCalculator(window=60)

    rolling = calc.calculate(result)

    assert len(rolling.timestamps) == 40


def test_zero_volatility():
    """Test handling of zero volatility."""
    start = datetime(2024, 1, 1)
    snapshots = []

    for i in range(60):
        ts = start + timedelta(days=i)
        snap = Snapshot(
            timestamp=ts,
            equity=100000.0,
            cash=100000.0,
            positions={},
            positions_value=0.0,
        )
        snapshots.append(snap)

    metrics = PerformanceMetrics(
        total_return=0.0,
        annualized_return=0.0,
        volatility=0.0,
        sharpe_ratio=0.0,
        sortino_ratio=0.0,
        max_drawdown=0.0,
        num_trades=0,
        win_rate=0.0,
        profit_factor=0.0,
        avg_trade_pnl=0.0,
        avg_leverage=0.0,
        max_leverage=0.0,
        avg_num_positions=0.0,
        start_date=start,
        end_date=start + timedelta(days=59),
        duration_days=60,
        daily_returns=[0.0] * 59,
    )

    result = BacktestResult(
        snapshots=snapshots,
        trades=[],
        metrics=metrics,
        strategy_name="Test",
        mode="backtest",
    )

    calc = RollingMetricsCalculator(window=30)
    rolling = calc.calculate(result)

    for sharpe in rolling.sharpe_ratio:
        assert sharpe == 0.0
