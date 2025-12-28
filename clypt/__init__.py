"""
Clypt Trading Engine - Alpha-factor based cryptocurrency trading system.

A production-ready quantitative trading engine for cryptocurrency markets
featuring alpha factor computation, portfolio optimization, and realistic
backtesting with proper look-ahead bias prevention.
"""

__version__ = "0.1.0"
__author__ = "Clypt Team"

from clypt.types import (
    # Enums
    EngineMode,
    FillStatus,
    OrderSide,
    OrderType,
    # Market Data
    OHLCV,
    Quote,
    # Trading
    Fill,
    Order,
    Position,
    # Portfolio
    Snapshot,
    # Performance
    BacktestResult,
    PerformanceMetrics,
    # Config
    Constraints,
    CostModel,
    EngineConfig,
)

__all__ = [
    # Enums
    "EngineMode",
    "FillStatus",
    "OrderSide",
    "OrderType",
    # Market Data
    "OHLCV",
    "Quote",
    # Trading
    "Fill",
    "Order",
    "Position",
    # Portfolio
    "Snapshot",
    # Performance
    "BacktestResult",
    "PerformanceMetrics",
    # Config
    "Constraints",
    "CostModel",
    "EngineConfig",
]
