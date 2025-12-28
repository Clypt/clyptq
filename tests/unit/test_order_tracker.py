"""Tests for order state tracking."""

from datetime import datetime, timedelta

from clypt.engine.order_tracker import OrderStatus, OrderTracker, TrackedOrder
from clypt.types import Fill, FillStatus, Order, OrderSide


def test_tracked_order_creation():
    order = Order(symbol="BTC/USDT", side=OrderSide.BUY, amount=1.0)
    tracked = TrackedOrder(order)

    assert tracked.status == OrderStatus.PENDING
    assert tracked.filled_amount == 0.0
    assert tracked.remaining_amount == 1.0
    assert not tracked.is_terminal


def test_tracked_order_submission():
    order = Order(symbol="BTC/USDT", side=OrderSide.BUY, amount=1.0)
    tracked = TrackedOrder(order)

    tracked.mark_submitted("exchange_order_123")

    assert tracked.status == OrderStatus.SUBMITTED
    assert tracked.exchange_order_id == "exchange_order_123"


def test_tracked_order_partial_fill():
    order = Order(symbol="BTC/USDT", side=OrderSide.BUY, amount=1.0)
    tracked = TrackedOrder(order)

    fill = Fill(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        amount=0.5,
        price=50000.0,
        fee=25.0,
        timestamp=datetime.utcnow(),
        status=FillStatus.PARTIAL,
    )

    tracked.add_fill(fill)

    assert tracked.status == OrderStatus.PARTIAL
    assert tracked.filled_amount == 0.5
    assert tracked.remaining_amount == 0.5
    assert not tracked.is_terminal


def test_tracked_order_full_fill():
    order = Order(symbol="BTC/USDT", side=OrderSide.BUY, amount=1.0)
    tracked = TrackedOrder(order)

    fill = Fill(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        amount=1.0,
        price=50000.0,
        fee=50.0,
        timestamp=datetime.utcnow(),
        status=FillStatus.FILLED,
    )

    tracked.add_fill(fill)

    assert tracked.status == OrderStatus.FILLED
    assert tracked.filled_amount == 1.0
    assert tracked.remaining_amount == 0.0
    assert tracked.is_terminal


def test_tracked_order_rejection():
    order = Order(symbol="BTC/USDT", side=OrderSide.BUY, amount=1.0)
    tracked = TrackedOrder(order)

    tracked.mark_rejected("Insufficient funds")

    assert tracked.status == OrderStatus.REJECTED
    assert tracked.error_message == "Insufficient funds"
    assert tracked.is_terminal


def test_tracked_order_cancellation():
    order = Order(symbol="BTC/USDT", side=OrderSide.BUY, amount=1.0)
    tracked = TrackedOrder(order)

    tracked.mark_cancelled()

    assert tracked.status == OrderStatus.CANCELLED
    assert tracked.is_terminal


def test_order_tracker_create_order():
    tracker = OrderTracker()
    order = Order(symbol="BTC/USDT", side=OrderSide.BUY, amount=1.0)

    tracked = tracker.create_order(order)

    assert tracked.order_id in tracker.orders
    assert tracked.order_id in tracker.orders_by_symbol["BTC/USDT"]


def test_order_tracker_get_order():
    tracker = OrderTracker()
    order = Order(symbol="BTC/USDT", side=OrderSide.BUY, amount=1.0)

    tracked = tracker.create_order(order)
    retrieved = tracker.get_order(tracked.order_id)

    assert retrieved is tracked


def test_order_tracker_get_pending_orders():
    tracker = OrderTracker()

    order1 = Order(symbol="BTC/USDT", side=OrderSide.BUY, amount=1.0)
    order2 = Order(symbol="ETH/USDT", side=OrderSide.BUY, amount=1.0)

    tracked1 = tracker.create_order(order1)
    tracked2 = tracker.create_order(order2)

    tracked1.mark_submitted("exchange_1")
    tracked2.mark_submitted("exchange_2")

    pending = tracker.get_pending_orders()
    assert len(pending) == 2

    tracked1.add_fill(
        Fill(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            amount=1.0,
            price=50000.0,
            fee=50.0,
            timestamp=datetime.utcnow(),
            status=FillStatus.FILLED,
        )
    )

    pending = tracker.get_pending_orders()
    assert len(pending) == 1
    assert pending[0] is tracked2


def test_order_tracker_get_pending_orders_by_symbol():
    tracker = OrderTracker()

    order1 = Order(symbol="BTC/USDT", side=OrderSide.BUY, amount=1.0)
    order2 = Order(symbol="ETH/USDT", side=OrderSide.BUY, amount=1.0)

    tracker.create_order(order1)
    tracker.create_order(order2)

    btc_pending = tracker.get_pending_orders("BTC/USDT")
    assert len(btc_pending) == 1
    assert btc_pending[0].order.symbol == "BTC/USDT"


def test_order_tracker_cleanup_old_orders():
    tracker = OrderTracker()

    order = Order(symbol="BTC/USDT", side=OrderSide.BUY, amount=1.0)
    tracked = tracker.create_order(order)

    tracked.add_fill(
        Fill(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            amount=1.0,
            price=50000.0,
            fee=50.0,
            timestamp=datetime.utcnow(),
            status=FillStatus.FILLED,
        )
    )

    tracked.updated_at = datetime.utcnow() - timedelta(days=2)

    assert len(tracker.orders) == 1
    tracker.cleanup_old_orders(max_age_seconds=86400)
    assert len(tracker.orders) == 0


def test_order_tracker_cleanup_keeps_recent():
    tracker = OrderTracker()

    order = Order(symbol="BTC/USDT", side=OrderSide.BUY, amount=1.0)
    tracked = tracker.create_order(order)

    tracked.add_fill(
        Fill(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            amount=1.0,
            price=50000.0,
            fee=50.0,
            timestamp=datetime.utcnow(),
            status=FillStatus.FILLED,
        )
    )

    tracker.cleanup_old_orders(max_age_seconds=86400)
    assert len(tracker.orders) == 1


def test_order_tracker_cleanup_keeps_pending():
    tracker = OrderTracker()

    order = Order(symbol="BTC/USDT", side=OrderSide.BUY, amount=1.0)
    tracked = tracker.create_order(order)

    tracked.updated_at = datetime.utcnow() - timedelta(days=2)

    tracker.cleanup_old_orders(max_age_seconds=86400)
    assert len(tracker.orders) == 1
