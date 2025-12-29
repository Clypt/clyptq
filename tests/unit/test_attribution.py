"""Tests for performance attribution."""

from datetime import datetime, timedelta

import pytest

from clyptq.analytics.attribution import PerformanceAttributor, AttributionResult
from clyptq.core.types import (
    BacktestResult,
    Snapshot,
    Position,
    Fill,
    PerformanceMetrics,
)


def create_test_backtest_result() -> BacktestResult:
    """Create simple backtest result for testing."""
    start = datetime(2024, 1, 1)

    snapshots = []
    for i in range(10):
        ts = start + timedelta(days=i)
        positions = {}

        if i > 0:
            positions["BTC/USDT"] = Position(
                symbol="BTC/USDT",
                amount=1.0,
                avg_price=40000.0 + i * 100,
            )

        positions_val = sum(p.amount * p.avg_price for p in positions.values())
        snap = Snapshot(
            timestamp=ts,
            equity=100000.0 + i * 500,
            cash=60000.0,
            positions=positions,
            positions_value=positions_val,
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
        total_return=0.045,
        annualized_return=0.16,
        volatility=0.12,
        sharpe_ratio=1.3,
        sortino_ratio=1.8,
        max_drawdown=0.05,
        num_trades=1,
        win_rate=1.0,
        profit_factor=2.0,
        avg_trade_pnl=500.0,
        avg_leverage=0.4,
        max_leverage=0.5,
        avg_num_positions=1.0,
        start_date=start,
        end_date=start + timedelta(days=9),
        duration_days=9,
        daily_returns=[0.005] * 9,
    )

    return BacktestResult(
        snapshots=snapshots,
        trades=fills,
        metrics=metrics,
        strategy_name="Test",
        mode="backtest",
    )


def test_attributor_initialization():
    """Test attributor initialization."""
    attributor = PerformanceAttributor()
    assert attributor is not None


def test_analyze_basic():
    """Test basic attribution analysis."""
    result = create_test_backtest_result()
    attributor = PerformanceAttributor()

    attribution = attributor.analyze(result)

    assert attribution is not None
    assert isinstance(attribution, AttributionResult)
    assert attribution.total_return > 0
    assert attribution.transaction_cost_drag < 0
    assert len(attribution.asset_attributions) > 0


def test_total_return_calculation():
    """Test total return calculation."""
    result = create_test_backtest_result()
    attributor = PerformanceAttributor()

    total_return = attributor._calculate_total_return(result)

    initial = result.snapshots[0].equity
    final = result.snapshots[-1].equity
    expected = (final - initial) / initial

    assert abs(total_return - expected) < 1e-6


def test_transaction_cost_drag():
    """Test transaction cost calculation."""
    result = create_test_backtest_result()
    attributor = PerformanceAttributor()

    cost_drag = attributor._calculate_transaction_cost_drag(result)

    assert cost_drag < 0
    total_fees = sum(fill.fee for fill in result.trades)
    expected = -total_fees / result.snapshots[0].equity
    assert abs(cost_drag - expected) < 1e-6


def test_cash_drag():
    """Test cash drag calculation."""
    result = create_test_backtest_result()
    attributor = PerformanceAttributor()

    cash_drag = attributor._calculate_cash_drag(result)

    assert cash_drag <= 0


def test_asset_attribution():
    """Test asset-level attribution."""
    result = create_test_backtest_result()
    attributor = PerformanceAttributor()

    asset_attr = attributor._calculate_asset_attribution(result)

    assert len(asset_attr) > 0
    assert asset_attr[0].symbol == "BTC/USDT"


def test_attribution_to_dict():
    """Test attribution result serialization."""
    result = create_test_backtest_result()
    attributor = PerformanceAttributor()

    attribution = attributor.analyze(result)
    attr_dict = attribution.to_dict()

    assert "total_return" in attr_dict
    assert "factor_attributions" in attr_dict
    assert "asset_attributions" in attr_dict
    assert "transaction_cost_drag" in attr_dict
    assert "cash_drag" in attr_dict
    assert "timestamp" in attr_dict

    assert isinstance(attr_dict["asset_attributions"], list)
    if attr_dict["asset_attributions"]:
        assert "symbol" in attr_dict["asset_attributions"][0]
        assert "total_return" in attr_dict["asset_attributions"][0]


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

    attributor = PerformanceAttributor()

    with pytest.raises(ValueError, match="No snapshots"):
        attributor.analyze(result)
