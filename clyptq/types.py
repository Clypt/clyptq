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


@dataclass
class BacktestResult:
    """Complete backtest results."""

    snapshots: List[Snapshot]  # Portfolio snapshots over time
    trades: List[Fill]  # All executed trades
    metrics: PerformanceMetrics  # Performance metrics
    strategy_name: str  # Name of the strategy
    mode: EngineMode  # Engine mode used

    def to_dict(self) -> Dict:
        """Export backtest results as dictionary for SaaS platform integration."""
        import json

        equity_curve = []
        for snapshot in self.snapshots:
            equity_curve.append({
                "timestamp": snapshot.timestamp.isoformat(),
                "equity": snapshot.equity,
                "cash": snapshot.cash,
                "positions_value": snapshot.positions_value,
                "leverage": snapshot.leverage,
                "num_positions": snapshot.num_positions,
            })

        trades_list = []
        for trade in self.trades:
            trades_list.append({
                "symbol": trade.symbol,
                "side": trade.side.value,
                "amount": trade.amount,
                "price": trade.price,
                "fee": trade.fee,
                "timestamp": trade.timestamp.isoformat(),
                "order_id": trade.order_id,
                "status": trade.status.value,
            })

        positions_timeline = []
        for snapshot in self.snapshots:
            positions_at_time = []
            for symbol, position in snapshot.positions.items():
                positions_at_time.append({
                    "symbol": symbol,
                    "amount": position.amount,
                    "avg_price": position.avg_price,
                    "unrealized_pnl": position.unrealized_pnl,
                    "realized_pnl": position.realized_pnl,
                })
            positions_timeline.append({
                "timestamp": snapshot.timestamp.isoformat(),
                "positions": positions_at_time,
            })

        drawdown_series = []
        peak = self.snapshots[0].equity if self.snapshots else 0.0
        for snapshot in self.snapshots:
            if snapshot.equity > peak:
                peak = snapshot.equity
            drawdown = (peak - snapshot.equity) / peak if peak > 0 else 0.0
            drawdown_series.append({
                "timestamp": snapshot.timestamp.isoformat(),
                "equity": snapshot.equity,
                "peak": peak,
                "drawdown": drawdown,
            })

        metrics_dict = {
            "total_return": self.metrics.total_return,
            "annualized_return": self.metrics.annualized_return,
            "volatility": self.metrics.volatility,
            "sharpe_ratio": self.metrics.sharpe_ratio,
            "sortino_ratio": self.metrics.sortino_ratio,
            "max_drawdown": self.metrics.max_drawdown,
            "num_trades": self.metrics.num_trades,
            "win_rate": self.metrics.win_rate,
            "profit_factor": self.metrics.profit_factor,
            "avg_trade_pnl": self.metrics.avg_trade_pnl,
            "avg_leverage": self.metrics.avg_leverage,
            "max_leverage": self.metrics.max_leverage,
            "avg_num_positions": self.metrics.avg_num_positions,
            "start_date": self.metrics.start_date.isoformat(),
            "end_date": self.metrics.end_date.isoformat(),
            "duration_days": self.metrics.duration_days,
        }

        return {
            "strategy_name": self.strategy_name,
            "mode": self.mode.value,
            "equity_curve": equity_curve,
            "trades": trades_list,
            "positions_timeline": positions_timeline,
            "drawdown_series": drawdown_series,
            "metrics": metrics_dict,
        }

    def to_json(self, indent: Optional[int] = 2) -> str:
        """Export backtest results as JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=indent)


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


@dataclass
class ExecutionResult:
    """Result from a single execution step (live/paper trading)."""

    timestamp: datetime
    action: str  # "rebalance" or "skip"
    fills: List[Fill]
    orders: List[Order]
    snapshot: Snapshot
    rebalance_reason: Optional[str] = None


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
