"""Tests for HTML report generation."""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from clyptq.analytics.reporting.report import HTMLReportGenerator
from clyptq.core.types import (
    BacktestResult,
    Snapshot,
    Fill,
    PerformanceMetrics,
)


def create_test_result() -> BacktestResult:
    """Create test backtest result."""
    start = datetime(2024, 1, 1)
    snapshots = []

    equity_values = (
        [100000 + i * 1000 for i in range(20)]
        + [119000 - i * 500 for i in range(10)]
        + [114000 + i * 800 for i in range(20)]
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

    fills = [
        Fill(
            timestamp=start,
            symbol="BTC/USDT",
            side="buy",
            amount=1.0,
            price=40000.0,
            fee=40.0,
        )
    ]

    metrics = PerformanceMetrics(
        total_return=0.3,
        annualized_return=0.45,
        volatility=0.15,
        sharpe_ratio=2.5,
        sortino_ratio=3.2,
        max_drawdown=0.08,
        num_trades=50,
        win_rate=0.62,
        profit_factor=2.1,
        avg_trade_pnl=500.0,
        avg_leverage=0.5,
        max_leverage=0.8,
        avg_num_positions=3.0,
        start_date=start,
        end_date=start + timedelta(days=len(equity_values) - 1),
        duration_days=len(equity_values),
        daily_returns=[0.01] * (len(equity_values) - 1),
    )

    return BacktestResult(
        snapshots=snapshots,
        trades=fills,
        metrics=metrics,
        strategy_name="TestStrategy",
        mode="backtest",
    )


def test_generator_initialization():
    """Test generator initialization."""
    gen = HTMLReportGenerator(rolling_window=30, min_drawdown=0.02)
    assert gen.rolling_window == 30
    assert gen.min_drawdown == 0.02


def test_generate_report():
    """Test basic report generation."""
    result = create_test_result()
    gen = HTMLReportGenerator()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        output_path = f.name

    try:
        gen.generate(result, output_path, title="Test Report")

        assert Path(output_path).exists()
        content = Path(output_path).read_text()

        assert "<!DOCTYPE html>" in content
        assert "Test Report" in content
        assert "TestStrategy" in content

    finally:
        Path(output_path).unlink()


def test_report_contains_metrics():
    """Test report contains performance metrics."""
    result = create_test_result()
    gen = HTMLReportGenerator()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        output_path = f.name

    try:
        gen.generate(result, output_path)
        content = Path(output_path).read_text()

        assert "Total Return" in content
        assert "Sharpe Ratio" in content
        assert "Max Drawdown" in content
        assert "Win Rate" in content

    finally:
        Path(output_path).unlink()


def test_report_contains_attribution():
    """Test report contains attribution section."""
    result = create_test_result()
    gen = HTMLReportGenerator()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        output_path = f.name

    try:
        gen.generate(result, output_path)
        content = Path(output_path).read_text()

        assert "Performance Attribution" in content
        assert "Transaction Costs" in content
        assert "Cash Drag" in content

    finally:
        Path(output_path).unlink()


def test_report_contains_drawdown():
    """Test report contains drawdown section."""
    result = create_test_result()
    gen = HTMLReportGenerator()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        output_path = f.name

    try:
        gen.generate(result, output_path)
        content = Path(output_path).read_text()

        assert "Drawdown Analysis" in content

    finally:
        Path(output_path).unlink()


def test_report_contains_rolling():
    """Test report contains rolling metrics."""
    result = create_test_result()
    gen = HTMLReportGenerator(rolling_window=30)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        output_path = f.name

    try:
        gen.generate(result, output_path)
        content = Path(output_path).read_text()

        assert "Rolling Metrics" in content

    finally:
        Path(output_path).unlink()


def test_css_styling():
    """Test CSS styling is included."""
    result = create_test_result()
    gen = HTMLReportGenerator()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        output_path = f.name

    try:
        gen.generate(result, output_path)
        content = Path(output_path).read_text()

        assert "<style>" in content
        assert "font-family" in content
        assert ".metric-card" in content

    finally:
        Path(output_path).unlink()


def test_default_title():
    """Test default title generation."""
    result = create_test_result()
    gen = HTMLReportGenerator()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        output_path = f.name

    try:
        gen.generate(result, output_path)
        content = Path(output_path).read_text()

        assert "TestStrategy Backtest Report" in content

    finally:
        Path(output_path).unlink()
