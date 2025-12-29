"""
HTML report generation example.

Demonstrates how to generate comprehensive backtest reports.
"""

from datetime import datetime, timedelta

from clyptq.analytics.report import HTMLReportGenerator
from clyptq.core.types import (
    BacktestResult,
    Snapshot,
    Fill,
    PerformanceMetrics,
)


def create_example_backtest() -> BacktestResult:
    """Create example backtest with realistic pattern."""
    start = datetime(2024, 1, 1)
    snapshots = []

    # Simulate realistic equity curve with growth and drawdowns
    equity_values = (
        [100000 + i * 800 for i in range(30)]  # Growth phase
        + [123200 - i * 600 for i in range(15)]  # Drawdown 1
        + [114200 + i * 900 for i in range(25)]  # Recovery + growth
        + [136700 - i * 400 for i in range(10)]  # Drawdown 2
        + [132700 + i * 1100 for i in range(30)]  # Final growth
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

    # Example trades
    fills = [
        Fill(
            timestamp=start + timedelta(days=5),
            symbol="BTC/USDT",
            side="buy",
            amount=1.0,
            price=42000.0,
            fee=42.0,
        ),
        Fill(
            timestamp=start + timedelta(days=20),
            symbol="ETH/USDT",
            side="buy",
            amount=10.0,
            price=2500.0,
            fee=25.0,
        ),
        Fill(
            timestamp=start + timedelta(days=50),
            symbol="SOL/USDT",
            side="buy",
            amount=100.0,
            price=100.0,
            fee=10.0,
        ),
    ]

    metrics = PerformanceMetrics(
        total_return=0.65,
        annualized_return=0.95,
        volatility=0.18,
        sharpe_ratio=4.2,
        sortino_ratio=6.1,
        max_drawdown=0.067,
        num_trades=42,
        win_rate=0.67,
        profit_factor=2.8,
        avg_trade_pnl=1547.0,
        avg_leverage=0.6,
        max_leverage=0.85,
        avg_num_positions=3.2,
        start_date=start,
        end_date=start + timedelta(days=len(equity_values) - 1),
        duration_days=len(equity_values),
        daily_returns=[0.006] * (len(equity_values) - 1),
    )

    return BacktestResult(
        snapshots=snapshots,
        trades=fills,
        metrics=metrics,
        strategy_name="Multi-Timeframe Momentum",
        mode="backtest",
    )


def main():
    print("=" * 70)
    print("HTML Report Generation Example")
    print("=" * 70)

    print("\n[1/3] Creating example backtest result...")
    result = create_example_backtest()
    print(f"  Total Return: {result.metrics.total_return:.2%}")
    print(f"  Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {result.metrics.max_drawdown:.2%}")

    print("\n[2/3] Generating HTML report...")
    generator = HTMLReportGenerator(
        rolling_window=30,
        min_drawdown=0.01,
    )

    output_path = "backtest_report.html"
    generator.generate(
        result=result,
        output_path=output_path,
        title="Multi-Timeframe Momentum Strategy Report",
    )

    print(f"  ✓ Report saved to: {output_path}")

    print("\n[3/3] Report sections:")
    print("  ✓ Summary (strategy, period, trades)")
    print("  ✓ Performance Metrics (returns, Sharpe, drawdown)")
    print("  ✓ Performance Attribution (costs, asset contributions)")
    print("  ✓ Drawdown Analysis (periods, duration, recovery)")
    print("  ✓ Rolling Metrics (30-day window)")

    print("\n" + "=" * 70)
    print(f"Open the report: open {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
