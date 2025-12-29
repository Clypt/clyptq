"""Test risk manager functionality."""

from clyptq.risk.manager import RiskManager
from clyptq.core.types import Order, OrderSide, Position


def test_position_size_limit_no_limit():
    rm = RiskManager()

    orders = [
        Order(symbol="BTC/USDT", side=OrderSide.BUY, amount=0.5),
        Order(symbol="ETH/USDT", side=OrderSide.BUY, amount=2.0),
    ]

    prices = {"BTC/USDT": 50000.0, "ETH/USDT": 3000.0}
    positions = {}
    equity = 10000.0

    result = rm.apply_position_limits(orders, positions, prices, equity)

    assert len(result) == 2
    assert result[0].amount == 0.5
    assert result[1].amount == 2.0


def test_position_size_limit_enforced():
    rm = RiskManager(max_position_pct=0.3)

    orders = [
        Order(symbol="BTC/USDT", side=OrderSide.BUY, amount=0.1),
    ]

    prices = {"BTC/USDT": 50000.0}
    positions = {}
    equity = 10000.0

    result = rm.apply_position_limits(orders, positions, prices, equity)

    assert len(result) == 1
    max_value = equity * 0.3
    max_amount = max_value / 50000.0
    assert abs(result[0].amount - max_amount) < 1e-6


def test_position_size_limit_with_existing_position():
    rm = RiskManager(max_position_pct=0.4)

    existing_pos = Position(
        symbol="BTC/USDT",
        amount=0.04,
        avg_price=50000.0,
        realized_pnl=0.0,
    )

    orders = [
        Order(symbol="BTC/USDT", side=OrderSide.BUY, amount=0.05),
    ]

    prices = {"BTC/USDT": 50000.0}
    positions = {"BTC/USDT": existing_pos}
    equity = 10000.0

    result = rm.apply_position_limits(orders, positions, prices, equity)

    assert len(result) == 1
    current_value = 0.04 * 50000.0
    max_value = equity * 0.4
    allowed_value = max_value - current_value
    allowed_amount = allowed_value / 50000.0
    assert abs(result[0].amount - allowed_amount) < 1e-6


def test_position_size_limit_sell_not_affected():
    rm = RiskManager(max_position_pct=0.3)

    orders = [
        Order(symbol="BTC/USDT", side=OrderSide.SELL, amount=0.1),
    ]

    prices = {"BTC/USDT": 50000.0}
    positions = {}
    equity = 10000.0

    result = rm.apply_position_limits(orders, positions, prices, equity)

    assert len(result) == 1
    assert result[0].amount == 0.1


def test_stop_loss():
    rm = RiskManager(stop_loss_pct=0.05)

    pos = Position(
        symbol="BTC/USDT",
        amount=0.1,
        avg_price=50000.0,
        realized_pnl=0.0,
    )

    positions = {"BTC/USDT": pos}
    prices = {"BTC/USDT": 47000.0}

    exit_orders = rm.check_position_exits(positions, prices)

    assert len(exit_orders) == 1
    assert exit_orders[0].symbol == "BTC/USDT"
    assert exit_orders[0].side == OrderSide.SELL
    assert exit_orders[0].amount == 0.1


def test_take_profit():
    rm = RiskManager(take_profit_pct=0.10)

    pos = Position(
        symbol="BTC/USDT",
        amount=0.1,
        avg_price=50000.0,
        realized_pnl=0.0,
    )

    positions = {"BTC/USDT": pos}
    prices = {"BTC/USDT": 56000.0}

    exit_orders = rm.check_position_exits(positions, prices)

    assert len(exit_orders) == 1
    assert exit_orders[0].symbol == "BTC/USDT"
    assert exit_orders[0].side == OrderSide.SELL


def test_max_drawdown():
    rm = RiskManager(max_drawdown_pct=0.15)

    rm.peak_equity = 10000.0
    assert not rm.check_max_drawdown(9000.0)

    assert rm.check_max_drawdown(8400.0)
