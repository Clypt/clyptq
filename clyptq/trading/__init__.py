"""Trading logic and business domain."""

from clyptq.trading.engine import BacktestEngine, LiveEngine
from clyptq.trading.execution import BacktestExecutor, LiveExecutor
from clyptq.strategy.base import Strategy
from clyptq.core.types import CostModel

# Operators module shortcut
from clyptq import operator

__all__ = [
    # Operators
    "operator",
    # Engines
    "BacktestEngine",
    "LiveEngine",
    # Executors
    "BacktestExecutor",
    "LiveExecutor",
    # Strategies
    "Strategy",
    # Cost
    "CostModel",
]
