"""Backtesting engine."""

from clyptq.engine.core import Engine

# Re-export for backward compatibility
from clyptq.execution import BacktestExecutor
from clyptq.execution.live import CCXTExecutor as LiveExecutor
from clyptq.portfolio.state import PortfolioState
from clyptq.risk import CostModel, RiskManager

__all__ = [
    "Engine",
    "BacktestExecutor",
    "LiveExecutor",
    "CCXTExecutor",
    "CostModel",
    "PortfolioState",
    "RiskManager",
]
