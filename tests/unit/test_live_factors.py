"""Test live factor computation with rolling buffer."""

from datetime import datetime, timedelta

import pytest

from clyptq.data.live.view import LiveDataView
from clyptq.data.live.buffer import RollingPriceBuffer
from clyptq.factors.library.momentum import MomentumFactor


def test_rolling_buffer_basic():
    buffer = RollingPriceBuffer(max_periods=10)

    prices = {"BTC/USDT": 50000.0, "ETH/USDT": 3000.0}
    timestamp = datetime(2023, 1, 1)

    buffer.update(timestamp, prices)

    assert len(buffer.timestamps) == 1
    assert "BTC/USDT" in buffer.prices
    assert "ETH/USDT" in buffer.prices


def test_rolling_buffer_lookback():
    buffer = RollingPriceBuffer(max_periods=100)

    base = datetime(2023, 1, 1)
    for i in range(50):
        timestamp = base + timedelta(days=i)
        prices = {"BTC/USDT": 50000.0 + i * 100}
        buffer.update(timestamp, prices)

    close_prices = buffer.get_close_prices("BTC/USDT", 20)
    assert close_prices is not None
    assert len(close_prices) == 20
    assert close_prices[-1] == 50000.0 + 49 * 100


def test_live_data_view():
    buffer = RollingPriceBuffer(max_periods=100)

    base = datetime(2023, 1, 1)
    for i in range(30):
        timestamp = base + timedelta(days=i)
        prices = {"BTC/USDT": 50000.0 + i * 100, "ETH/USDT": 3000.0 + i * 10}
        buffer.update(timestamp, prices)

    view = LiveDataView(buffer, base + timedelta(days=29))

    assert "BTC/USDT" in view.symbols
    assert "ETH/USDT" in view.symbols

    btc_prices = view.close("BTC/USDT", 10)
    assert btc_prices is not None
    assert len(btc_prices) == 10

    current = view.current_price("BTC/USDT")
    assert current == 50000.0 + 29 * 100


def test_momentum_factor_with_live_view():
    buffer = RollingPriceBuffer(max_periods=100)

    base = datetime(2023, 1, 1)
    for i in range(30):
        timestamp = base + timedelta(days=i)
        prices = {
            "BTC/USDT": 50000.0 + i * 100,
            "ETH/USDT": 3000.0 - i * 10,
        }
        buffer.update(timestamp, prices)

    view = LiveDataView(buffer, base + timedelta(days=29))
    factor = MomentumFactor(lookback=20)

    scores = factor.compute(view)

    assert "BTC/USDT" in scores
    assert "ETH/USDT" in scores
    assert scores["BTC/USDT"] > scores["ETH/USDT"]


def test_warmup_period_check():
    buffer = RollingPriceBuffer(max_periods=100)

    base = datetime(2023, 1, 1)
    for i in range(5):
        timestamp = base + timedelta(days=i)
        prices = {"BTC/USDT": 50000.0}
        buffer.update(timestamp, prices)

    assert len(buffer.timestamps) == 5
    assert not buffer.has_sufficient_data("BTC/USDT", 20)

    for i in range(5, 25):
        timestamp = base + timedelta(days=i)
        prices = {"BTC/USDT": 50000.0}
        buffer.update(timestamp, prices)

    assert buffer.has_sufficient_data("BTC/USDT", 20)
