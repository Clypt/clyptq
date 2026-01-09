"""
Core types and primitives for ClyptQ.
"""

# Re-export from new locations for backwards compatibility
from clyptq.trading.execution.base import Executor
from clyptq.strategy.base import Strategy
from clyptq.core.clock import (
    BacktestClock,
    Clock,
    LiveClock,
)
from clyptq.core.timeframe import (
    calculate_system_clock,
    timeframe_to_minutes,
    minutes_to_timeframe,
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
    "Strategy",
    # Clock
    "Clock",
    "BacktestClock",
    "LiveClock",
    # Timeframe
    "calculate_system_clock",
    "timeframe_to_minutes",
    "minutes_to_timeframe",
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
