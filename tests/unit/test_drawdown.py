"""Tests for drawdown analysis."""

from datetime import datetime, timedelta

import pytest

from clyptq.analytics.drawdown import DrawdownAnalyzer, DrawdownAnalysis, DrawdownPeriod
from clyptq.core.types import (
    BacktestResult,
    Snapshot,
    PerformanceMetrics,
)


def create_test_result_with_drawdown() -> BacktestResult:
    """Create test result with drawdown pattern."""
    start = datetime(2024, 1, 1)
    snapshots = []

    equity_values = (
        [100000 + i * 1000 for i in range(10)]
        + [109000 - i * 500 for i in range(10)]
        + [104000 + i * 1000 for i in range(10)]
        + [113000 - i * 300 for i in range(5)]
    )

    for i, equity in enumerate(equity_values):
        ts = start + timedelta(days=i)
        snap = Snapshot(
            timestamp=ts,
            equity=equity,
            cash=50000.0,
            positions={},
            positions_value=0.0,
        )
        snapshots.append(snap)

    metrics = PerformanceMetrics(
        total_return=0.1,
        annualized_return=0.15,
        volatility=0.12,
        sharpe_ratio=1.0,
        sortino_ratio=1.5,
        max_drawdown=0.05,
        num_trades=10,
        win_rate=0.6,
        profit_factor=1.5,
        avg_trade_pnl=100.0,
        avg_leverage=0.5,
        max_leverage=0.8,
        avg_num_positions=2.0,
        start_date=start,
        end_date=start + timedelta(days=len(equity_values) - 1),
        duration_days=len(equity_values),
        daily_returns=[0.01] * (len(equity_values) - 1),
    )

    return BacktestResult(
        snapshots=snapshots,
        trades=[],
        metrics=metrics,
        strategy_name="Test",
        mode="backtest",
    )


def test_analyzer_initialization():
    """Test analyzer initialization."""
    analyzer = DrawdownAnalyzer(min_drawdown=0.02)
    assert analyzer.min_drawdown == 0.02


def test_analyze_drawdown():
    """Test basic drawdown analysis."""
    result = create_test_result_with_drawdown()
    analyzer = DrawdownAnalyzer(min_drawdown=0.01)

    analysis = analyzer.analyze(result)

    assert isinstance(analysis, DrawdownAnalysis)
    assert analysis.max_drawdown > 0
    assert analysis.avg_drawdown > 0
    assert len(analysis.drawdown_periods) > 0
    assert len(analysis.underwater_equity) == len(result.snapshots)
    assert len(analysis.timestamps) == len(result.snapshots)


def test_drawdown_period_structure():
    """Test drawdown period structure."""
    result = create_test_result_with_drawdown()
    analyzer = DrawdownAnalyzer(min_drawdown=0.01)

    analysis = analyzer.analyze(result)

    for period in analysis.drawdown_periods:
        assert isinstance(period, DrawdownPeriod)
        assert period.start <= period.end
        assert period.depth < 0
        assert period.duration_days > 0

        if period.recovery:
            assert period.recovery >= period.end
            assert period.recovery_days is not None


def test_sorted_by_depth():
    """Test periods sorted by depth."""
    result = create_test_result_with_drawdown()
    analyzer = DrawdownAnalyzer(min_drawdown=0.01)

    analysis = analyzer.analyze(result)

    if len(analysis.drawdown_periods) > 1:
        for i in range(len(analysis.drawdown_periods) - 1):
            assert (
                abs(analysis.drawdown_periods[i].depth)
                >= abs(analysis.drawdown_periods[i + 1].depth)
            )


def test_underwater_equity():
    """Test underwater equity calculation."""
    result = create_test_result_with_drawdown()
    analyzer = DrawdownAnalyzer()

    analysis = analyzer.analyze(result)

    for val in analysis.underwater_equity:
        assert val <= 0


def test_min_drawdown_filter():
    """Test minimum drawdown filtering."""
    result = create_test_result_with_drawdown()

    analyzer_strict = DrawdownAnalyzer(min_drawdown=0.05)
    analysis_strict = analyzer_strict.analyze(result)

    analyzer_loose = DrawdownAnalyzer(min_drawdown=0.01)
    analysis_loose = analyzer_loose.analyze(result)

    assert len(analysis_strict.drawdown_periods) <= len(analysis_loose.drawdown_periods)


def test_empty_snapshots_error():
    """Test error on empty snapshots."""
    result = BacktestResult(
        snapshots=[],
        trades=[],
        metrics=PerformanceMetrics(
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
            start_date=datetime.now(),
            end_date=datetime.now(),
            duration_days=0,
            daily_returns=[],
        ),
        strategy_name="Test",
        mode="backtest",
    )

    analyzer = DrawdownAnalyzer()

    with pytest.raises(ValueError, match="No snapshots"):
        analyzer.analyze(result)


def test_to_dict():
    """Test result serialization."""
    result = create_test_result_with_drawdown()
    analyzer = DrawdownAnalyzer(min_drawdown=0.01)

    analysis = analyzer.analyze(result)
    data = analysis.to_dict()

    assert "max_drawdown" in data
    assert "avg_drawdown" in data
    assert "drawdown_periods" in data
    assert "underwater_equity" in data
    assert "timestamps" in data

    assert isinstance(data["drawdown_periods"], list)
    if data["drawdown_periods"]:
        period = data["drawdown_periods"][0]
        assert "start" in period
        assert "end" in period
        assert "depth" in period
        assert "duration_days" in period


def test_no_drawdown():
    """Test with no drawdown (always increasing)."""
    start = datetime(2024, 1, 1)
    snapshots = []

    for i in range(30):
        ts = start + timedelta(days=i)
        snap = Snapshot(
            timestamp=ts,
            equity=100000.0 + i * 1000,
            cash=50000.0,
            positions={},
            positions_value=0.0,
        )
        snapshots.append(snap)

    metrics = PerformanceMetrics(
        total_return=0.3,
        annualized_return=0.4,
        volatility=0.1,
        sharpe_ratio=2.0,
        sortino_ratio=3.0,
        max_drawdown=0.0,
        num_trades=5,
        win_rate=1.0,
        profit_factor=5.0,
        avg_trade_pnl=500.0,
        avg_leverage=0.5,
        max_leverage=0.8,
        avg_num_positions=2.0,
        start_date=start,
        end_date=start + timedelta(days=29),
        duration_days=30,
        daily_returns=[0.01] * 29,
    )

    result = BacktestResult(
        snapshots=snapshots,
        trades=[],
        metrics=metrics,
        strategy_name="Test",
        mode="backtest",
    )

    analyzer = DrawdownAnalyzer()
    analysis = analyzer.analyze(result)

    assert analysis.max_drawdown == 0.0
    assert len(analysis.drawdown_periods) == 0
