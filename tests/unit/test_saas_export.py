"""Test SaaS export functionality."""

from datetime import datetime, timedelta
import json

from clyptq.core.types import (
    BacktestResult,
    EngineMode,
    Fill,
    FillStatus,
    OrderSide,
    PerformanceMetrics,
    Position,
    Snapshot,
)


def test_backtest_result_export():
    """Test BacktestResult.to_dict() and to_json() methods."""
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 5)

    snapshots = [
        Snapshot(
            timestamp=start_date + timedelta(days=i),
            equity=10000 + i * 100,
            cash=5000 - i * 50,
            positions={"BTC/USDT": Position("BTC/USDT", 0.1, 50000.0, 100.0, 50.0)},
            positions_value=5000 + i * 150,
        )
        for i in range(5)
    ]

    trades = [
        Fill(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            amount=0.1,
            price=50000.0,
            fee=5.0,
            timestamp=start_date,
            order_id="order_1",
            status=FillStatus.FILLED,
        ),
        Fill(
            symbol="ETH/USDT",
            side=OrderSide.BUY,
            amount=2.0,
            price=3000.0,
            fee=6.0,
            timestamp=start_date + timedelta(days=1),
            order_id="order_2",
            status=FillStatus.FILLED,
        ),
    ]

    metrics = PerformanceMetrics(
        total_return=0.05,
        annualized_return=0.15,
        daily_returns=[0.01, 0.02, -0.01, 0.03],
        volatility=0.12,
        sharpe_ratio=1.25,
        sortino_ratio=1.5,
        max_drawdown=0.08,
        num_trades=2,
        win_rate=0.6,
        profit_factor=2.5,
        avg_trade_pnl=50.0,
        avg_leverage=0.5,
        max_leverage=0.8,
        avg_num_positions=1.2,
        start_date=start_date,
        end_date=end_date,
        duration_days=4,
    )

    result = BacktestResult(
        snapshots=snapshots,
        trades=trades,
        metrics=metrics,
        strategy_name="MomentumStrategy",
        mode=EngineMode.BACKTEST,
    )

    # Test to_dict()
    result_dict = result.to_dict()
    assert result_dict["strategy_name"] == "MomentumStrategy"
    assert result_dict["mode"] == "backtest"
    assert len(result_dict["equity_curve"]) == 5
    assert len(result_dict["trades"]) == 2

    # Test to_json()
    result_json = result.to_json(indent=2)
    parsed = json.loads(result_json)

    assert parsed["strategy_name"] == "MomentumStrategy"
    assert parsed["mode"] == "backtest"
    assert len(parsed["equity_curve"]) == 5
    assert len(parsed["trades"]) == 2


if __name__ == "__main__":
    test_backtest_result_export()
