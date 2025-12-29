"""Order state tracking for live trading."""

import uuid
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from clyptq.types import Fill, Order


class OrderStatus(Enum):
    """Order lifecycle states."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class TrackedOrder:
    """Order with state tracking."""

    def __init__(self, order: Order, order_id: Optional[str] = None):
        self.order_id = order_id or str(uuid.uuid4())
        self.order = order
        self.status = OrderStatus.PENDING
        self.filled_amount = 0.0
        self.fills: List[Fill] = []
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
        self.exchange_order_id: Optional[str] = None
        self.error_message: Optional[str] = None

    def add_fill(self, fill: Fill) -> None:
        """Add a fill and update state."""
        self.fills.append(fill)
        self.filled_amount += fill.amount
        self.updated_at = datetime.now(timezone.utc)

        if self.filled_amount >= self.order.amount:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIAL

    def mark_submitted(self, exchange_order_id: Optional[str] = None) -> None:
        self.status = OrderStatus.SUBMITTED
        self.exchange_order_id = exchange_order_id
        self.updated_at = datetime.now(timezone.utc)

    def mark_rejected(self, error: str) -> None:
        self.status = OrderStatus.REJECTED
        self.error_message = error
        self.updated_at = datetime.now(timezone.utc)

    def mark_cancelled(self) -> None:
        self.status = OrderStatus.CANCELLED
        self.updated_at = datetime.now(timezone.utc)

    @property
    def remaining_amount(self) -> float:
        return max(0.0, self.order.amount - self.filled_amount)

    @property
    def is_terminal(self) -> bool:
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]


class OrderTracker:
    """Tracks order lifecycle for live trading."""

    def __init__(self):
        self.orders: Dict[str, TrackedOrder] = {}
        self.orders_by_symbol: Dict[str, List[str]] = defaultdict(list)

    def create_order(self, order: Order, order_id: Optional[str] = None) -> TrackedOrder:
        tracked = TrackedOrder(order, order_id)
        self.orders[tracked.order_id] = tracked
        self.orders_by_symbol[order.symbol].append(tracked.order_id)
        return tracked

    def get_order(self, order_id: str) -> Optional[TrackedOrder]:
        return self.orders.get(order_id)

    def get_pending_orders(self, symbol: Optional[str] = None) -> List[TrackedOrder]:
        if symbol:
            order_ids = self.orders_by_symbol.get(symbol, [])
            return [
                self.orders[oid] for oid in order_ids
                if not self.orders[oid].is_terminal
            ]
        return [o for o in self.orders.values() if not o.is_terminal]

    def get_all_orders(self, symbol: Optional[str] = None) -> List[TrackedOrder]:
        if symbol:
            order_ids = self.orders_by_symbol.get(symbol, [])
            return [self.orders[oid] for oid in order_ids]
        return list(self.orders.values())

    def cleanup_old_orders(self, max_age_seconds: int = 86400) -> None:
        """Remove old terminal orders."""
        now = datetime.now(timezone.utc)
        to_remove = []

        for order_id, tracked in self.orders.items():
            if tracked.is_terminal:
                age = (now - tracked.updated_at).total_seconds()
                if age > max_age_seconds:
                    to_remove.append(order_id)

        for order_id in to_remove:
            tracked = self.orders.pop(order_id)
            self.orders_by_symbol[tracked.order.symbol].remove(order_id)

    def clear(self) -> None:
        self.orders.clear()
        self.orders_by_symbol.clear()
