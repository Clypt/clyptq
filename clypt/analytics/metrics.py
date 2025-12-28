"""
Performance metrics calculation.

Computes comprehensive performance metrics from backtest results.

CRITICAL FIX: Auto-detects period frequency instead of hardcoding 365.
"""

from typing import List

import numpy as np

from clypt.types import Fill, PerformanceMetrics, Snapshot


def auto_detect_periods_per_year(snapshots: List[Snapshot]) -> int:
    """
    Auto-detect periods per year from snapshot frequency.

    CRITICAL: Prevents hardcoding 365 assumption.

    Args:
        snapshots: List of portfolio snapshots

    Returns:
        Estimated periods per year

    Examples:
        - 1-minute bars: 365 * 24 * 60 = 525,600
        - 1-hour bars: 365 * 24 = 8,760
        - Daily bars: 365
        - Weekly bars: 52
        - Monthly bars: 12
    """
    if len(snapshots) < 10:
        # Not enough data, assume daily
        return 365

    # Sample first 10 time gaps
    time_diffs = [
        (snapshots[i + 1].timestamp - snapshots[i].timestamp).total_seconds()
        for i in range(min(10, len(snapshots) - 1))
    ]

    avg_seconds = np.mean(time_diffs)

    # Detect frequency
    if avg_seconds < 120:  # < 2 minutes
        # Minute bars
        return 365 * 24 * 60
    elif avg_seconds < 7200:  # < 2 hours
        # Hourly bars
        return 365 * 24
    elif avg_seconds < 86400 * 2:  # < 2 days
        # Daily bars
        return 365
    elif avg_seconds < 86400 * 10:  # < 10 days
        # Weekly bars
        return 52
    else:
        # Monthly bars
        return 12


def compute_metrics(snapshots: List[Snapshot], trades: List[Fill]) -> PerformanceMetrics:
    """
    Compute all performance metrics from snapshots and trades.

    AUTO-DETECTS period frequency from data instead of hardcoding.

    Args:
        snapshots: Portfolio snapshots over time
        trades: All executed trades

    Returns:
        PerformanceMetrics with all metrics calculated
    """
    if not snapshots:
        # Return default metrics if no snapshots (e.g., short test period with weekly schedule)
        return PerformanceMetrics(
            total_return=0.0,
            annualized_return=0.0,
            daily_returns=[],
            volatility=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            num_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            avg_trade_pnl=0.0,
            max_position_concentration=0.0,
            avg_num_positions=0.0,
        )

    # Auto-detect periods per year
    periods_per_year = auto_detect_periods_per_year(snapshots)

    # Extract equity series
    equity_series = np.array([s.equity for s in snapshots])
    initial_equity = equity_series[0]
    final_equity = equity_series[-1]

    # Calculate returns
    total_return = (final_equity - initial_equity) / initial_equity if initial_equity > 0 else 0.0

    # Daily returns
    daily_returns = []
    for i in range(1, len(equity_series)):
        ret = (equity_series[i] - equity_series[i - 1]) / equity_series[i - 1]
        daily_returns.append(ret)

    daily_returns_array = np.array(daily_returns) if daily_returns else np.array([0.0])

    # Annualized return
    num_periods = len(snapshots)
    years = num_periods / periods_per_year if periods_per_year > 0 else 1.0

    if years > 0 and total_return > -1:
        annualized_return = (1 + total_return) ** (1 / years) - 1
    else:
        annualized_return = 0.0

    # Volatility
    volatility = np.std(daily_returns_array) * np.sqrt(periods_per_year)

    # Sharpe ratio (assume 0% risk-free rate)
    if volatility > 1e-10:
        sharpe_ratio = annualized_return / volatility
    else:
        sharpe_ratio = 0.0

    # Sortino ratio (downside deviation)
    negative_returns = daily_returns_array[daily_returns_array < 0]
    if len(negative_returns) > 0:
        downside_std = np.std(negative_returns)
        downside_volatility = downside_std * np.sqrt(periods_per_year)
        if downside_volatility > 1e-10:
            sortino_ratio = annualized_return / downside_volatility
        else:
            sortino_ratio = 0.0
    else:
        sortino_ratio = 0.0

    # Maximum drawdown
    peak = equity_series[0]
    max_dd = 0.0

    for equity in equity_series:
        if equity > peak:
            peak = equity

        drawdown = (peak - equity) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, drawdown)

    # Trading metrics
    num_trades = len(trades)

    # Win rate and profit factor
    if num_trades > 0:
        trade_pnls = []
        for i, trade in enumerate(trades):
            # Simplified P&L calculation
            if trade.side.value == "buy":
                # For buys, we'll calculate P&L when sold
                continue
            else:
                # For sells, calculate P&L
                # Find corresponding buy
                buy_price = None
                for j in range(i - 1, -1, -1):
                    if (
                        trades[j].symbol == trade.symbol
                        and trades[j].side.value == "buy"
                    ):
                        buy_price = trades[j].price
                        break

                if buy_price:
                    pnl = (trade.price - buy_price) * trade.amount - trade.fee
                    trade_pnls.append(pnl)

        if trade_pnls:
            winning_trades = [p for p in trade_pnls if p > 0]
            losing_trades = [p for p in trade_pnls if p < 0]

            win_rate = len(winning_trades) / len(trade_pnls) if trade_pnls else 0.0

            gross_profit = sum(winning_trades) if winning_trades else 0.0
            gross_loss = abs(sum(losing_trades)) if losing_trades else 0.0

            if gross_loss > 1e-10:
                profit_factor = gross_profit / gross_loss
            else:
                profit_factor = 0.0 if gross_profit == 0 else float("inf")

            avg_trade_pnl = np.mean(trade_pnls)
        else:
            win_rate = 0.0
            profit_factor = 0.0
            avg_trade_pnl = 0.0
    else:
        win_rate = 0.0
        profit_factor = 0.0
        avg_trade_pnl = 0.0

    # Exposure metrics
    leverage_series = [s.leverage for s in snapshots]
    num_positions_series = [s.num_positions for s in snapshots]

    avg_leverage = np.mean(leverage_series) if leverage_series else 0.0
    max_leverage = np.max(leverage_series) if leverage_series else 0.0
    avg_num_positions = np.mean(num_positions_series) if num_positions_series else 0.0

    # Time metrics
    start_date = snapshots[0].timestamp
    end_date = snapshots[-1].timestamp
    duration_days = (end_date - start_date).days

    return PerformanceMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        daily_returns=daily_returns,
        volatility=volatility,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        max_drawdown=max_dd,
        num_trades=num_trades,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_trade_pnl=avg_trade_pnl,
        avg_leverage=avg_leverage,
        max_leverage=max_leverage,
        avg_num_positions=avg_num_positions,
        start_date=start_date,
        end_date=end_date,
        duration_days=duration_days,
    )


def print_metrics(metrics: PerformanceMetrics) -> None:
    """
    Print performance metrics in readable format.

    Args:
        metrics: PerformanceMetrics to print
    """
    print("\n" + "=" * 60)
    print("PERFORMANCE METRICS")
    print("=" * 60)

    print(f"\n📊 Returns:")
    print(f"  Total Return:       {metrics.total_return:>10.2%}")
    print(f"  Annualized Return:  {metrics.annualized_return:>10.2%}")

    print(f"\n⚡ Risk:")
    print(f"  Volatility:         {metrics.volatility:>10.2%}")
    print(f"  Sharpe Ratio:       {metrics.sharpe_ratio:>10.2f}")
    print(f"  Sortino Ratio:      {metrics.sortino_ratio:>10.2f}")
    print(f"  Max Drawdown:       {metrics.max_drawdown:>10.2%}")

    print(f"\n💰 Trading:")
    print(f"  Number of Trades:   {metrics.num_trades:>10}")
    print(f"  Win Rate:           {metrics.win_rate:>10.2%}")
    print(f"  Profit Factor:      {metrics.profit_factor:>10.2f}")
    print(f"  Avg Trade P&L:      ${metrics.avg_trade_pnl:>9.2f}")

    print(f"\n📈 Exposure:")
    print(f"  Avg Leverage:       {metrics.avg_leverage:>10.2f}x")
    print(f"  Max Leverage:       {metrics.max_leverage:>10.2f}x")
    print(f"  Avg Positions:      {metrics.avg_num_positions:>10.1f}")

    print(f"\n📅 Duration:")
    print(f"  Start:              {metrics.start_date.strftime('%Y-%m-%d')}")
    print(f"  End:                {metrics.end_date.strftime('%Y-%m-%d')}")
    print(f"  Days:               {metrics.duration_days:>10}")

    print("=" * 60 + "\n")
