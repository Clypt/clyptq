"""
Unit tests for DataStore - Critical Test 1: Look-ahead Bias Prevention

Tests that available_symbols() does not use future delisting information.
"""

from datetime import datetime

import pandas as pd
import pytest

from clyptq.data.store import DataStore


def test_available_symbols_no_lookahead():
    """
    CRITICAL TEST 1: Look-ahead bias prevention.

    data/store.py:207-217 - MUST NOT use future delisting info.

    Tests that available_symbols() only returns symbols that:
    1. Were already listed by the query timestamp
    2. Have at least one data point at or before the query timestamp

    Does NOT consider when symbols will be delisted (future information).
    """
    store = DataStore()

    # BTC starts on Jan 1, 2023
    btc_data = pd.DataFrame({
        "open": [100.0, 101.0, 102.0],
        "high": [101.0, 102.0, 103.0],
        "low": [99.0, 100.0, 101.0],
        "close": [100.5, 101.5, 102.5],
        "volume": [1000.0, 1100.0, 1200.0],
    }, index=pd.DatetimeIndex([
        datetime(2023, 1, 1),
        datetime(2023, 1, 2),
        datetime(2023, 1, 3),
    ]))

    # ETH starts on June 1, 2023 (much later)
    eth_data = pd.DataFrame({
        "open": [200.0, 201.0],
        "high": [201.0, 202.0],
        "low": [199.0, 200.0],
        "close": [200.5, 201.5],
        "volume": [2000.0, 2100.0],
    }, index=pd.DatetimeIndex([
        datetime(2023, 6, 1),
        datetime(2023, 6, 2),
    ]))

    store.add_ohlcv("BTC/USDT", btc_data)
    store.add_ohlcv("ETH/USDT", eth_data)

    # Test 1: March 1 - BTC should be available, ETH should NOT
    available_march = store.available_symbols(datetime(2023, 3, 1))

    assert "BTC/USDT" in available_march, "BTC should be available (was listed before March)"
    assert "ETH/USDT" not in available_march, "ETH should NOT be available (not listed until June)"

    # Test 2: June 1 - Both should be available
    available_june = store.available_symbols(datetime(2023, 6, 1))

    assert "BTC/USDT" in available_june
    assert "ETH/USDT" in available_june

    # Test 3: Before BTC listing - Nothing available
    available_early = store.available_symbols(datetime(2022, 12, 31))

    assert len(available_early) == 0, "No symbols should be available before any listings"


def test_available_symbols_edge_cases():
    """Test edge cases for available_symbols()."""
    store = DataStore()

    # Add symbol with single data point
    single_data = pd.DataFrame({
        "open": [100.0],
        "high": [101.0],
        "low": [99.0],
        "close": [100.5],
        "volume": [1000.0],
    }, index=pd.DatetimeIndex([datetime(2023, 1, 1)]))

    store.add_ohlcv("SINGLE/USDT", single_data)

    # Exactly at listing time
    available = store.available_symbols(datetime(2023, 1, 1))
    assert "SINGLE/USDT" in available

    # One second before listing
    available_before = store.available_symbols(datetime(2022, 12, 31, 23, 59, 59))
    assert "SINGLE/USDT" not in available_before


def test_available_symbols_delisting():
    """Test that delisted symbols are not available after delisting date."""
    store = DataStore()

    # Delisted coin (YFII-style: ends at 2023-08-22)
    delisted_data = pd.DataFrame({
        "open": [100.0, 101.0, 102.0],
        "high": [101.0, 102.0, 103.0],
        "low": [99.0, 100.0, 101.0],
        "close": [100.5, 101.5, 102.5],
        "volume": [1000.0, 1100.0, 1200.0],
    }, index=pd.DatetimeIndex([
        datetime(2023, 8, 20),
        datetime(2023, 8, 21),
        datetime(2023, 8, 22),
    ]))

    # Active coin
    active_data = pd.DataFrame({
        "open": [200.0, 201.0, 202.0, 203.0],
        "high": [201.0, 202.0, 203.0, 204.0],
        "low": [199.0, 200.0, 201.0, 202.0],
        "close": [200.5, 201.5, 202.5, 203.5],
        "volume": [2000.0, 2100.0, 2200.0, 2300.0],
    }, index=pd.DatetimeIndex([
        datetime(2023, 8, 20),
        datetime(2023, 8, 21),
        datetime(2023, 8, 22),
        datetime(2023, 8, 23),
    ]))

    store.add_ohlcv("YFII/USDT", delisted_data)
    store.add_ohlcv("BTC/USDT", active_data)

    # On last trading day - both available
    available_aug22 = store.available_symbols(datetime(2023, 8, 22))
    assert "YFII/USDT" in available_aug22
    assert "BTC/USDT" in available_aug22

    # After delisting - only active coin available
    available_aug23 = store.available_symbols(datetime(2023, 8, 23))
    assert "YFII/USDT" not in available_aug23, "Delisted coin should NOT be available"
    assert "BTC/USDT" in available_aug23

    # Few days later - delisted coin still not available
    available_future = store.available_symbols(datetime(2023, 8, 25))
    assert "YFII/USDT" not in available_future
    assert "BTC/USDT" not in available_future  # BTC data also ended at Aug 23


def test_dataview_temporal_consistency():
    """Test that DataView only provides historical data."""
    store = DataStore()

    # Create data spanning multiple days
    data = pd.DataFrame({
        "open": [100.0, 101.0, 102.0, 103.0, 104.0],
        "high": [101.0, 102.0, 103.0, 104.0, 105.0],
        "low": [99.0, 100.0, 101.0, 102.0, 103.0],
        "close": [100.5, 101.5, 102.5, 103.5, 104.5],
        "volume": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
    }, index=pd.DatetimeIndex([
        datetime(2023, 1, 1),
        datetime(2023, 1, 2),
        datetime(2023, 1, 3),
        datetime(2023, 1, 4),
        datetime(2023, 1, 5),
    ]))

    store.add_ohlcv("BTC/USDT", data)

    # Get view at Jan 3
    view = store.get_view(datetime(2023, 1, 3))

    # Should be able to get data up to Jan 3
    prices = view.close("BTC/USDT", lookback=3)
    assert len(prices) == 3
    assert prices[-1] == 102.5  # Jan 3 close

    # Should NOT be able to see Jan 4 or Jan 5 data
    with pytest.raises(ValueError, match="Insufficient data"):
        view.close("BTC/USDT", lookback=5)


def test_dataview_current_price():
    """Test that current_price returns the latest available price."""
    store = DataStore()

    data = pd.DataFrame({
        "open": [100.0, 101.0, 102.0],
        "high": [101.0, 102.0, 103.0],
        "low": [99.0, 100.0, 101.0],
        "close": [100.5, 101.5, 102.5],
        "volume": [1000.0, 1100.0, 1200.0],
    }, index=pd.DatetimeIndex([
        datetime(2023, 1, 1),
        datetime(2023, 1, 2),
        datetime(2023, 1, 3),
    ]))

    store.add_ohlcv("BTC/USDT", data)

    # View at Jan 2
    view = store.get_view(datetime(2023, 1, 2))
    price = view.current_price("BTC/USDT")

    assert price == 101.5, "Should return Jan 2 close, not future data"

    # View at Jan 3
    view_jan3 = store.get_view(datetime(2023, 1, 3))
    price_jan3 = view_jan3.current_price("BTC/USDT")

    assert price_jan3 == 102.5


def test_datastore_metadata():
    """Test DataStore metadata tracking."""
    store = DataStore()

    data = pd.DataFrame({
        "open": [100.0, 101.0],
        "high": [101.0, 102.0],
        "low": [99.0, 100.0],
        "close": [100.5, 101.5],
        "volume": [1000.0, 1100.0],
    }, index=pd.DatetimeIndex([
        datetime(2023, 1, 1),
        datetime(2023, 1, 2),
    ]))

    store.add_ohlcv("BTC/USDT", data, frequency="1d", source="test")

    # Check metadata
    metadata = store.get_metadata("BTC/USDT")

    assert metadata.symbol == "BTC/USDT"
    assert metadata.num_bars == 2
    assert metadata.frequency == "1d"
    assert metadata.source == "test"
    assert metadata.start_date == datetime(2023, 1, 1)
    assert metadata.end_date == datetime(2023, 1, 2)


def test_datastore_operations():
    """Test basic DataStore operations."""
    store = DataStore()

    # Initially empty
    assert len(store) == 0
    assert "BTC/USDT" not in store

    # Add data
    data = pd.DataFrame({
        "open": [100.0],
        "high": [101.0],
        "low": [99.0],
        "close": [100.5],
        "volume": [1000.0],
    }, index=pd.DatetimeIndex([datetime(2023, 1, 1)]))

    store.add_ohlcv("BTC/USDT", data)

    # Check operations
    assert len(store) == 1
    assert "BTC/USDT" in store
    assert store.has_symbol("BTC/USDT")
    assert "BTC/USDT" in store.symbols()

    # Remove symbol
    store.remove_symbol("BTC/USDT")
    assert len(store) == 0
    assert "BTC/USDT" not in store

    # Clear
    store.add_ohlcv("BTC/USDT", data)
    store.add_ohlcv("ETH/USDT", data)
    assert len(store) == 2

    store.clear()
    assert len(store) == 0
