"""
Analytics module for clyptq.

Two categories:
1. Performance analytics (for Engine backtest results)
   - compute_metrics: Backtest performance metrics
   - PerformanceAttributor: Return attribution
   - RollingMetricsCalculator: Rolling performance metrics
   - DrawdownAnalyzer: Drawdown period analysis
   - MonteCarloSimulator: Monte Carlo simulation

2. Summary statistics (for final analysis)
   - ic: Information Coefficient
   - sharpe: Sharpe ratio
   - sortino: Sortino ratio
   - max_drawdown: Maximum drawdown

Note: For most analytical operations, use clyptq.operator.
      analytics is for final statistical summaries.
"""

# Performance analytics (Engine output)
from clyptq.analytics.performance import (
    compute_metrics,
    PerformanceAttributor,
    RollingMetricsCalculator,
    DrawdownAnalyzer,
)
from clyptq.analytics.risk import MonteCarloSimulator

# Summary statistics
from clyptq.analytics.summary import (
    ic,
    ic_summary,
    sharpe,
    sortino,
    calmar,
    max_drawdown,
    var,
    cvar,
)

__all__ = [
    # Performance (Engine)
    "compute_metrics",
    "PerformanceAttributor",
    "RollingMetricsCalculator",
    "DrawdownAnalyzer",
    "MonteCarloSimulator",
    # Summary stats
    "ic",
    "ic_summary",
    "sharpe",
    "sortino",
    "calmar",
    "max_drawdown",
    "var",
    "cvar",
]
