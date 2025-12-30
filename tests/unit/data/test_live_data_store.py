"""Tests for LiveDataStore."""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from clyptq.data.stores.live_store import LiveDataStore


def test_add_historical():
    store = LiveDataStore(lookback_days=30)

    df = pd.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1) + timedelta(days=i) for i in range(50)
            ],
            "open": [100.0 + i for i in range(50)],
            "high": [101.0 + i for i in range(50)],
            "low": [99.0 + i for i in range(50)],
            "close": [100.5 + i for i in range(50)],
            "volume": [1000.0 for _ in range(50)],
        }
    )

    store.add_historical("BTC/USDT", df)

    assert store.has_symbol("BTC/USDT")
    assert len(store.data["BTC/USDT"]) <= 30


def test_update():
    store = LiveDataStore(lookback_days=10)

    now = datetime(2024, 1, 15, 12, 0)
    prices = {"BTC/USDT": 45000.0, "ETH/USDT": 2500.0}

    store.update(now, prices)

    assert store.has_symbol("BTC/USDT")
    assert store.has_symbol("ETH/USDT")
    assert len(store.data["BTC/USDT"]) == 1
    assert store.data["BTC/USDT"].iloc[-1]["close"] == 45000.0


def test_rolling_window():
    store = LiveDataStore(lookback_days=5)

    base = datetime(2024, 1, 1)
    for i in range(10):
        ts = base + timedelta(days=i)
        prices = {"BTC/USDT": 40000.0 + i * 100}
        store.update(ts, prices)

    assert len(store.data["BTC/USDT"]) <= 5


def test_get_view():
    store = LiveDataStore(lookback_days=30)

    df = pd.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(30)],
            "open": [100.0 for _ in range(30)],
            "high": [101.0 for _ in range(30)],
            "low": [99.0 for _ in range(30)],
            "close": [100.0 + i for i in range(30)],
            "volume": [1000.0 for _ in range(30)],
        }
    )

    store.add_historical("BTC/USDT", df)

    view = store.get_view(datetime(2024, 1, 15))

    assert "BTC/USDT" in view.symbols
    assert view.current_price("BTC/USDT") == 114.0


def test_available_symbols():
    store = LiveDataStore(lookback_days=60)

    df_btc = pd.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(40)],
            "open": [100.0 for _ in range(40)],
            "high": [101.0 for _ in range(40)],
            "low": [99.0 for _ in range(40)],
            "close": [100.0 for _ in range(40)],
            "volume": [1000.0 for _ in range(40)],
        }
    )

    df_eth = pd.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(10)],
            "open": [50.0 for _ in range(10)],
            "high": [51.0 for _ in range(10)],
            "low": [49.0 for _ in range(10)],
            "close": [50.0 for _ in range(10)],
            "volume": [500.0 for _ in range(10)],
        }
    )

    store.add_historical("BTC/USDT", df_btc)
    store.add_historical("ETH/USDT", df_eth)

    symbols = store.available_symbols(datetime(2024, 1, 25), min_bars=20)

    assert "BTC/USDT" in symbols
    assert "ETH/USDT" not in symbols


def test_reset():
    store = LiveDataStore()

    df = pd.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1)],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
            "volume": [1000.0],
        }
    )

    store.add_historical("BTC/USDT", df)
    assert store.num_symbols() == 1

    store.reset()
    assert store.num_symbols() == 0
