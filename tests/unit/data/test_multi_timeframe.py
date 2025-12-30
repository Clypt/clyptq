"""Tests for multi-timeframe data store and factors."""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from clyptq.core.base import MultiTimeframeFactor
from clyptq.data.stores.mtf_store import MultiTimeframeStore
from clyptq.factors.library.momentum import MultiTimeframeMomentum


def create_test_data() -> MultiTimeframeStore:
    """Create multi-timeframe test data."""
    mtf_store = MultiTimeframeStore()

    # Create 1h data
    start = datetime(2023, 1, 1)
    hours = pd.date_range(start=start, periods=168, freq="h")  # 7 days
    prices_1h = [100.0 + i * 0.5 for i in range(168)]

    data_1h = pd.DataFrame(
        {
            "open": prices_1h,
            "high": [p * 1.01 for p in prices_1h],
            "low": [p * 0.99 for p in prices_1h],
            "close": [p * 1.005 for p in prices_1h],
            "volume": [1000.0] * 168,
        },
        index=hours,
    )

    mtf_store.add_ohlcv("BTC/USDT", data_1h, "1h")

    # Create 1d data
    days = pd.date_range(start=start, periods=30, freq="D")
    prices_1d = [100.0 + i * 2.0 for i in range(30)]

    data_1d = pd.DataFrame(
        {
            "open": prices_1d,
            "high": [p * 1.02 for p in prices_1d],
            "low": [p * 0.98 for p in prices_1d],
            "close": [p * 1.01 for p in prices_1d],
            "volume": [5000.0] * 30,
        },
        index=days,
    )

    mtf_store.add_ohlcv("BTC/USDT", data_1d, "1d")

    # Create 1w data
    weeks = pd.date_range(start=start, periods=8, freq="W")
    prices_1w = [100.0 + i * 10.0 for i in range(8)]

    data_1w = pd.DataFrame(
        {
            "open": prices_1w,
            "high": [p * 1.05 for p in prices_1w],
            "low": [p * 0.95 for p in prices_1w],
            "close": [p * 1.03 for p in prices_1w],
            "volume": [25000.0] * 8,
        },
        index=weeks,
    )

    mtf_store.add_ohlcv("BTC/USDT", data_1w, "1w")

    return mtf_store


def test_mtf_store_initialization():
    """Test MTF store initialization."""
    mtf_store = MultiTimeframeStore()

    # Check all timeframes initialized
    assert len(mtf_store.stores) == 4
    assert "1h" in mtf_store.stores
    assert "4h" in mtf_store.stores
    assert "1d" in mtf_store.stores
    assert "1w" in mtf_store.stores


def test_add_ohlcv_valid_timeframes():
    """Test adding data for valid timeframes."""
    mtf_store = MultiTimeframeStore()

    data = pd.DataFrame(
        {
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1000.0],
        },
        index=[datetime(2023, 1, 1)],
    )

    # Should work for all valid timeframes
    for tf in ["1h", "4h", "1d", "1w"]:
        mtf_store.add_ohlcv("BTC/USDT", data, tf)
        assert "BTC/USDT" in mtf_store.stores[tf]._data


def test_add_ohlcv_invalid_timeframe():
    """Test error on invalid timeframe."""
    mtf_store = MultiTimeframeStore()

    data = pd.DataFrame(
        {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5], "volume": [1000.0]},
        index=[datetime(2023, 1, 1)],
    )

    with pytest.raises(ValueError, match="Invalid timeframe"):
        mtf_store.add_ohlcv("BTC/USDT", data, "5m")


def test_get_store():
    """Test getting store for specific timeframe."""
    mtf_store = create_test_data()

    store_1d = mtf_store.get_store("1d")
    assert "BTC/USDT" in store_1d._data

    with pytest.raises(ValueError):
        mtf_store.get_store("invalid")


def test_available_timeframes():
    """Test getting available timeframes for symbol."""
    mtf_store = create_test_data()

    available = mtf_store.available_timeframes("BTC/USDT")
    assert "1h" in available
    assert "1d" in available
    assert "1w" in available

    # Symbol with no data
    available_empty = mtf_store.available_timeframes("ETH/USDT")
    assert len(available_empty) == 0


def test_resample_1h_to_1d():
    """Test resampling from 1h to 1d."""
    mtf_store = create_test_data()

    resampled = mtf_store.resample_to_timeframe("BTC/USDT", "1h", "1d")

    assert resampled is not None
    assert len(resampled) == 7  # 7 days from 168 hours

    # Check OHLCV aggregation
    assert "open" in resampled.columns
    assert "high" in resampled.columns
    assert "low" in resampled.columns
    assert "close" in resampled.columns
    assert "volume" in resampled.columns


def test_resample_1d_to_1w():
    """Test resampling from 1d to 1w."""
    mtf_store = create_test_data()

    resampled = mtf_store.resample_to_timeframe("BTC/USDT", "1d", "1w")

    assert resampled is not None
    assert len(resampled) >= 4  # At least 4 weeks from 30 days


def test_resample_downsampling_not_supported():
    """Test that downsampling returns None."""
    mtf_store = create_test_data()

    # Cannot downsample 1d -> 1h
    resampled = mtf_store.resample_to_timeframe("BTC/USDT", "1d", "1h")
    assert resampled is None


def test_align_timestamps():
    """Test timestamp alignment to timeframe boundaries."""
    mtf_store = MultiTimeframeStore()

    # Test timestamp: 2023-01-05 14:30:00
    timestamp = datetime(2023, 1, 5, 14, 30, 0)
    aligned = mtf_store.align_timestamps(timestamp, ["1h", "4h", "1d", "1w"])

    # 1h: Should floor to 14:00:00
    assert aligned["1h"] == datetime(2023, 1, 5, 14, 0, 0)

    # 4h: Should floor to 12:00:00 (14 // 4 = 3, 3 * 4 = 12)
    assert aligned["4h"] == datetime(2023, 1, 5, 12, 0, 0)

    # 1d: Should floor to 00:00:00
    assert aligned["1d"] == datetime(2023, 1, 5, 0, 0, 0)

    # 1w: Should floor to Monday
    # 2023-01-05 is Thursday, so Monday is 2023-01-02
    assert aligned["1w"] == datetime(2023, 1, 2, 0, 0, 0)


def test_get_bar_at_timestamp():
    """Test getting bar at specific timestamp."""
    mtf_store = create_test_data()

    # Get 1d bar at 2023-01-05
    timestamp = datetime(2023, 1, 5, 12, 0, 0)
    bar = mtf_store.get_bar_at_timestamp("BTC/USDT", "1d", timestamp)

    assert bar is not None
    assert "close" in bar.index
    assert bar["close"] > 0


def test_get_bar_no_data():
    """Test getting bar when no data available."""
    mtf_store = create_test_data()

    # Try to get bar before data starts
    timestamp = datetime(2022, 12, 1)
    bar = mtf_store.get_bar_at_timestamp("BTC/USDT", "1d", timestamp)

    assert bar is None


def test_has_sufficient_data():
    """Test checking for sufficient data."""
    mtf_store = create_test_data()

    # Should have enough 1d data for 20-day lookback
    timestamp = datetime(2023, 1, 25)
    assert mtf_store.has_sufficient_data("BTC/USDT", "1d", timestamp, 20)

    # Should not have enough for 50-day lookback
    assert not mtf_store.has_sufficient_data("BTC/USDT", "1d", timestamp, 50)

    # Symbol with no data
    assert not mtf_store.has_sufficient_data("ETH/USDT", "1d", timestamp, 20)


def test_mtf_factor_initialization():
    """Test multi-timeframe factor initialization."""
    factor = MultiTimeframeMomentum(
        timeframes=["1d", "1w"],
        lookbacks={"1d": 20, "1w": 12},
    )

    assert factor.timeframes == ["1d", "1w"]
    assert factor.lookbacks["1d"] == 20
    assert factor.lookbacks["1w"] == 12
    assert factor.warmup_periods() == 20  # Max lookback


def test_mtf_factor_invalid_timeframe():
    """Test error on invalid timeframe."""
    with pytest.raises(ValueError, match="Invalid timeframe"):
        MultiTimeframeMomentum(timeframes=["5m", "1d"])


def test_mtf_momentum_compute():
    """Test multi-timeframe momentum computation."""
    mtf_store = create_test_data()

    factor = MultiTimeframeMomentum(
        timeframes=["1d", "1w"],
        lookbacks={"1d": 10, "1w": 3},  # Use 3 weeks (3 bars available by 2023-01-20)
        weights={"1d": 0.6, "1w": 0.4},
    )

    timestamp = datetime(2023, 1, 20)
    scores = factor.compute(mtf_store, timestamp, ["BTC/USDT"])

    assert "BTC/USDT" in scores
    assert scores["BTC/USDT"] > 0  # Uptrending data


def test_mtf_momentum_weights_validation():
    """Test that weights must sum to 1.0."""
    with pytest.raises(ValueError, match="sum to 1.0"):
        MultiTimeframeMomentum(
            timeframes=["1d", "1w"],
            weights={"1d": 0.5, "1w": 0.6},  # Sum = 1.1
        )


def test_mtf_factor_insufficient_data():
    """Test factor returns empty when insufficient data."""
    mtf_store = create_test_data()

    factor = MultiTimeframeMomentum(
        timeframes=["1d", "1w"],
        lookbacks={"1d": 100, "1w": 50},  # Too much lookback
    )

    timestamp = datetime(2023, 1, 20)
    scores = factor.compute(mtf_store, timestamp, ["BTC/USDT"])

    assert len(scores) == 0  # No scores due to insufficient data


def test_mtf_factor_required_timeframes():
    """Test getting required timeframes."""
    factor = MultiTimeframeMomentum(timeframes=["1h", "1d", "1w"])

    required = factor.required_timeframes()
    assert required == ["1h", "1d", "1w"]
