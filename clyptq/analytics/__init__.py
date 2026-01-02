"""Performance analytics and metrics."""

from clyptq.analytics.factors import FactorAnalyzer
from clyptq.analytics.performance import (
    compute_metrics,
    PerformanceAttributor,
    RollingMetricsCalculator,
    DrawdownAnalyzer,
)
from clyptq.analytics.risk import MonteCarloSimulator
from clyptq.analytics.reporting import HTMLReportGenerator, DataExplorer

__all__ = [
    "FactorAnalyzer",
    "compute_metrics",
    "PerformanceAttributor",
    "RollingMetricsCalculator",
    "DrawdownAnalyzer",
    "MonteCarloSimulator",
    "HTMLReportGenerator",
    "DataExplorer",
    "HistoricalSimulator",
]


def __getattr__(name):
    if name == "HistoricalSimulator":
        from clyptq.analytics.simulation import HistoricalSimulator

        return HistoricalSimulator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
