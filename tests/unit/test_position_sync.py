"""Tests for position synchronization."""

from clyptq.execution.positions.synchronizer import PositionDiscrepancy, PositionSynchronizer
from clyptq.types import Position


def test_position_discrepancy():
    disc = PositionDiscrepancy(
        symbol="BTC/USDT",
        internal_amount=1.0,
        exchange_amount=1.5,
        internal_avg_price=50000.0,
        exchange_avg_price=51000.0,
    )

    assert disc.amount_diff == 0.5
    assert disc.is_critical


def test_position_discrepancy_not_critical():
    disc = PositionDiscrepancy(
        symbol="BTC/USDT",
        internal_amount=1.0,
        exchange_amount=1.0000001,
        internal_avg_price=50000.0,
        exchange_avg_price=50000.0,
    )

    assert not disc.is_critical


def test_check_discrepancies_none():
    sync = PositionSynchronizer()

    internal = {"BTC/USDT": Position("BTC/USDT", 1.0, 50000.0, 0.0)}
    exchange = {"BTC/USDT": {"amount": 1.0, "avg_price": 50000.0}}

    discrepancies = sync.check_discrepancies(internal, exchange)
    assert len(discrepancies) == 0


def test_check_discrepancies_amount_diff():
    sync = PositionSynchronizer()

    internal = {"BTC/USDT": Position("BTC/USDT", 1.0, 50000.0, 0.0)}
    exchange = {"BTC/USDT": {"amount": 1.5, "avg_price": 50000.0}}

    discrepancies = sync.check_discrepancies(internal, exchange)
    assert len(discrepancies) == 1
    assert discrepancies[0].symbol == "BTC/USDT"
    assert discrepancies[0].amount_diff == 0.5


def test_check_discrepancies_missing_internal():
    sync = PositionSynchronizer()

    internal = {}
    exchange = {"BTC/USDT": {"amount": 1.0, "avg_price": 50000.0}}

    discrepancies = sync.check_discrepancies(internal, exchange)
    assert len(discrepancies) == 1
    assert discrepancies[0].internal_amount == 0.0
    assert discrepancies[0].exchange_amount == 1.0


def test_check_discrepancies_missing_exchange():
    sync = PositionSynchronizer()

    internal = {"BTC/USDT": Position("BTC/USDT", 1.0, 50000.0, 0.0)}
    exchange = {}

    discrepancies = sync.check_discrepancies(internal, exchange)
    assert len(discrepancies) == 1
    assert discrepancies[0].internal_amount == 1.0
    assert discrepancies[0].exchange_amount == 0.0


def test_check_discrepancies_within_tolerance():
    sync = PositionSynchronizer(tolerance=0.01)

    internal = {"BTC/USDT": Position("BTC/USDT", 1.0, 50000.0, 0.0)}
    exchange = {"BTC/USDT": {"amount": 1.005, "avg_price": 50000.0}}

    discrepancies = sync.check_discrepancies(internal, exchange)
    assert len(discrepancies) == 0


def test_sync_positions():
    sync = PositionSynchronizer()

    internal = {"BTC/USDT": Position("BTC/USDT", 1.0, 50000.0, 0.0)}
    exchange = {"ETH/USDT": {"amount": 2.0, "avg_price": 3000.0}}

    synced = sync.sync_positions(internal, exchange)

    assert "BTC/USDT" not in synced
    assert "ETH/USDT" in synced
    assert synced["ETH/USDT"].amount == 2.0
    assert synced["ETH/USDT"].avg_price == 3000.0


def test_sync_positions_skips_zero():
    sync = PositionSynchronizer()

    internal = {}
    exchange = {
        "BTC/USDT": {"amount": 1.0, "avg_price": 50000.0},
        "ETH/USDT": {"amount": 0.0, "avg_price": 0.0},
    }

    synced = sync.sync_positions(internal, exchange)

    assert "BTC/USDT" in synced
    assert "ETH/USDT" not in synced


def test_reconcile_position_none_to_position():
    sync = PositionSynchronizer()

    reconciled = sync.reconcile_position(
        internal_pos=None, exchange_amount=1.0, exchange_avg_price=50000.0
    )

    assert reconciled is not None
    assert reconciled.amount == 1.0
    assert reconciled.avg_price == 50000.0


def test_reconcile_position_zero_amount():
    sync = PositionSynchronizer()

    reconciled = sync.reconcile_position(
        internal_pos=None, exchange_amount=0.0, exchange_avg_price=0.0
    )

    assert reconciled is None


def test_reconcile_position_no_change():
    sync = PositionSynchronizer()

    internal_pos = Position("BTC/USDT", 1.0, 50000.0, 0.0)

    reconciled = sync.reconcile_position(
        internal_pos=internal_pos, exchange_amount=1.0, exchange_avg_price=50000.0
    )

    assert reconciled is internal_pos


def test_reconcile_position_update():
    sync = PositionSynchronizer()

    internal_pos = Position("BTC/USDT", 1.0, 50000.0, 0.0, 100.0)

    reconciled = sync.reconcile_position(
        internal_pos=internal_pos, exchange_amount=1.5, exchange_avg_price=51000.0
    )

    assert reconciled is not internal_pos
    assert reconciled.symbol == "BTC/USDT"
    assert reconciled.amount == 1.5
    assert reconciled.avg_price == 51000.0
    assert reconciled.realized_pnl == 100.0


def test_reconcile_position_within_tolerance():
    sync = PositionSynchronizer(tolerance=0.01)

    internal_pos = Position("BTC/USDT", 1.0, 50000.0, 0.0)

    reconciled = sync.reconcile_position(
        internal_pos=internal_pos, exchange_amount=1.005, exchange_avg_price=50000.0
    )

    assert reconciled is internal_pos
