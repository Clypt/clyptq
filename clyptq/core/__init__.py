"""
Core types and primitives for ClyptQ.
"""

from clyptq.core.base import (
    Executor,
    Factor,
    MultiTimeframeFactor,
    PortfolioConstructor,
    Strategy,
)
from clyptq.core.types import (
    BacktestResult,
    Constraints,
    CostModel,
    EngineMode,
    ExecutionResult,
    Fill,
    FillStatus,
    MonteCarloResult,
    OHLCV,
    Order,
    OrderSide,
    OrderType,
    PerformanceMetrics,
    Position,
    Quote,
    Snapshot,
)

__all__ = [
    # Base classes
    "Executor",
    "Factor",
    "MultiTimeframeFactor",
    "PortfolioConstructor",
    "Strategy",
    # Enums
    "EngineMode",
    "OrderSide",
    "OrderType",
    "FillStatus",
    # Market data
    "OHLCV",
    "Quote",
    # Trading primitives
    "Order",
    "Fill",
    "Position",
    # Portfolio
    "Snapshot",
    # Configuration
    "Constraints",
    "CostModel",
    # Results
    "BacktestResult",
    "PerformanceMetrics",
    "ExecutionResult",
    "MonteCarloResult",
]
