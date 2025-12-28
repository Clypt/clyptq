"""
Core type definitions for the Clypt Trading Engine.

This module contains all fundamental data structures used throughout the engine:
- Market data types (OHLCV, quotes)
- Trading primitives (orders, fills, positions)
- Portfolio state tracking
- Performance metrics and snapshots
- Configuration and constraints
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


# ============================================================================
# Enums
# ============================================================================


class EngineMode(Enum):
    """Engine execution mode."""

    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


class OrderSide(Enum):
    """Order side (buy or sell)."""

    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type."""

    MARKET = "market"
    LIMIT = "limit"


class FillStatus(Enum):
    """Fill status."""

    FILLED = "filled"
    PARTIAL = "partial"
    REJECTED = "rejected"


# ============================================================================
# Market Data Types
# ============================================================================


@dataclass
class OHLCV:
    """OHLCV candlestick data."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str


@dataclass
class Quote:
    """Real-time price quote."""

    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    last: float
    volume: float


# ============================================================================
# Trading Primitives
# ============================================================================


@dataclass
class Order:
    """Order to be executed."""

    symbol: str
    side: OrderSide
    amount: float  # In base currency (e.g., BTC for BTC/USDT)
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    timestamp: Optional[datetime] = None


@dataclass
class Fill:
    """Executed fill."""

    symbol: str
    side: OrderSide
    amount: float  # Positive for both buy and sell
    price: float  # Execution price
    fee: float  # Trading fee in quote currency
    timestamp: datetime
    order_id: Optional[str] = None
    status: FillStatus = FillStatus.FILLED


@dataclass
class Position:
    """Current position in a symbol."""

    symbol: str
    amount: float  # Positive for long, negative for short
    avg_price: float  # Average entry price
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


# ============================================================================
# Portfolio State
# ============================================================================


@dataclass
class Snapshot:
    """Portfolio snapshot at a point in time."""

    timestamp: datetime
    equity: float  # Total portfolio value (cash + positions)
    cash: float  # Available cash
    positions: Dict[str, Position]  # {symbol: Position}
    positions_value: float  # Total value of all positions
    leverage: float = 0.0  # positions_value / equity
    num_positions: int = 0  # Number of active positions

    def __post_init__(self) -> None:
        """Calculate derived fields."""
        self.num_positions = len([p for p in self.positions.values() if abs(p.amount) > 1e-8])
        if self.equity > 1e-8:
            self.leverage = self.positions_value / self.equity


# ============================================================================
# Performance Metrics
# ============================================================================


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""

    # Returns
    total_return: float  # (final - initial) / initial
    annualized_return: float  # Annualized compound return
    daily_returns: List[float]  # Daily returns series

    # Risk
    volatility: float  # Annualized volatility
    sharpe_ratio: float  # Risk-adjusted return
    sortino_ratio: float  # Downside risk-adjusted return
    max_drawdown: float  # Maximum peak-to-trough decline

    # Trading
    num_trades: int
    win_rate: float  # Percentage of profitable trades
    profit_factor: float  # Gross profit / gross loss
    avg_trade_pnl: float  # Average P&L per trade

    # Exposure
    avg_leverage: float  # Average leverage
    max_leverage: float  # Maximum leverage reached
    avg_num_positions: float  # Average number of positions

    # Time
    start_date: datetime
    end_date: datetime
    duration_days: int


# ============================================================================
# Backtest Results
# ============================================================================


@dataclass
class BacktestResult:
    """Complete backtest results."""

    snapshots: List[Snapshot]  # Portfolio snapshots over time
    trades: List[Fill]  # All executed trades
    metrics: PerformanceMetrics  # Performance metrics
    strategy_name: str  # Name of the strategy
    mode: EngineMode  # Engine mode used


# ============================================================================
# Configuration Types
# ============================================================================


@dataclass
class Constraints:
    """Portfolio construction constraints."""

    max_position_size: float = 0.2  # Max 20% per position
    max_gross_exposure: float = 1.0  # Max 100% invested
    min_position_size: float = 0.01  # Min 1% per position
    max_num_positions: int = 20  # Max number of positions
    allow_short: bool = False  # Allow short positions


@dataclass
class CostModel:
    """Trading cost model."""

    maker_fee: float = 0.001  # 0.1% maker fee
    taker_fee: float = 0.001  # 0.1% taker fee
    slippage_bps: float = 5.0  # 5 bps slippage


@dataclass
class EngineConfig:
    """Engine configuration."""

    mode: EngineMode
    initial_capital: float = 10000.0
    cost_model: CostModel = field(default_factory=CostModel)
    constraints: Constraints = field(default_factory=Constraints)
    rebalance_schedule: str = "daily"  # "daily", "weekly", "monthly"


# ============================================================================
# Strategy Interface Types
# ============================================================================


@dataclass
class FactorExposure:
    """Factor exposure for a symbol."""

    symbol: str
    score: float  # Factor score
    raw_score: Optional[float] = None  # Pre-normalized score
    rank: Optional[int] = None  # Rank among universe


@dataclass
class SignalEvent:
    """Trading signal event."""

    timestamp: datetime
    scores: Dict[str, float]  # {symbol: score}
    target_weights: Dict[str, float]  # {symbol: weight}
    current_weights: Dict[str, float]  # {symbol: weight}
    rebalance_needed: bool = True


# ============================================================================
# Data Store Types
# ============================================================================


@dataclass
class DataRange:
    """Date range for data queries."""

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        """Validate date range."""
        if self.start >= self.end:
            raise ValueError(f"start ({self.start}) must be before end ({self.end})")


@dataclass
class DataMetadata:
    """Metadata for stored data."""

    symbol: str
    start_date: datetime
    end_date: datetime
    num_bars: int
    frequency: str  # "1m", "5m", "1h", "1d", etc.
    source: str = "unknown"


# ============================================================================
# Cache Types
# ============================================================================


@dataclass
class CacheEntry:
    """Cache entry for factor computations."""

    timestamp: datetime
    data: Dict[str, float]  # {symbol: score}
    hash_key: str  # Hash of input data
    hit_count: int = 0
    last_accessed: Optional[datetime] = None


@dataclass
class CacheStats:
    """Cache performance statistics."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size: int = 0
    hit_rate: float = 0.0

    def update_hit_rate(self) -> None:
        """Calculate current hit rate."""
        total = self.hits + self.misses
        self.hit_rate = self.hits / total if total > 0 else 0.0
