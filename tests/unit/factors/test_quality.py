import numpy as np
import pandas as pd
from datetime import datetime

from clyptq.data.stores.store import DataStore
from clyptq.trading.factors.library.quality import (
    MarketDepthProxyFactor,
    PriceImpactFactor,
    VolumeStabilityFactor,
)


def create_test_store():
    store = DataStore()
    dates = pd.date_range(start=datetime(2024, 1, 1), periods=30, freq="D")

    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(30) * 0.5)

    stable_volume = pd.DataFrame(
        {
            "open": prices,
            "high": prices + 1,
            "low": prices - 1,
            "close": prices,
            "volume": [10000 + np.random.randn() * 100 for _ in range(30)],
        },
        index=dates,
    )

    volatile_volume = pd.DataFrame(
        {
            "open": prices,
            "high": prices + 1,
            "low": prices - 1,
            "close": prices,
            "volume": [10000 + np.random.randn() * 5000 for _ in range(30)],
        },
        index=dates,
    )

    store.add_ohlcv("STABLE", stable_volume)
    store.add_ohlcv("VOLATILE", volatile_volume)

    return store


def test_volume_stability_basic():
    store = create_test_store()
    factor = VolumeStabilityFactor(lookback=20)

    view = store.get_view(datetime(2024, 1, 30))
    scores = factor.compute(view)

    assert "STABLE" in scores
    assert "VOLATILE" in scores


def test_volume_stability_prefers_stable():
    store = create_test_store()
    factor = VolumeStabilityFactor(lookback=20)

    view = store.get_view(datetime(2024, 1, 30))
    scores = factor.compute(view)

    assert scores["STABLE"] > scores["VOLATILE"]


def test_price_impact_basic():
    store = DataStore()
    dates = pd.date_range(start=datetime(2024, 1, 1), periods=30, freq="D")

    np.random.seed(42)
    prices_high_volume = 100 + np.cumsum(np.random.randn(30) * 0.5)
    prices_low_volume = 100 + np.cumsum(np.random.randn(30) * 0.5)

    high_volume = pd.DataFrame(
        {
            "open": prices_high_volume,
            "high": prices_high_volume + 1,
            "low": prices_high_volume - 1,
            "close": prices_high_volume,
            "volume": [100000] * 30,
        },
        index=dates,
    )

    low_volume = pd.DataFrame(
        {
            "open": prices_low_volume,
            "high": prices_low_volume + 1,
            "low": prices_low_volume - 1,
            "close": prices_low_volume,
            "volume": [1000] * 30,
        },
        index=dates,
    )

    store.add_ohlcv("HIGH_VOL", high_volume)
    store.add_ohlcv("LOW_VOL", low_volume)

    factor = PriceImpactFactor(lookback=20)
    view = store.get_view(datetime(2024, 1, 30))
    scores = factor.compute(view)

    assert "HIGH_VOL" in scores
    assert "LOW_VOL" in scores


def test_price_impact_prefers_low_impact():
    store = DataStore()
    dates = pd.date_range(start=datetime(2024, 1, 1), periods=30, freq="D")

    np.random.seed(42)
    prices_high_volume = 100 + np.cumsum(np.random.randn(30) * 0.5)
    prices_low_volume = 100 + np.cumsum(np.random.randn(30) * 0.5)

    high_volume = pd.DataFrame(
        {
            "open": prices_high_volume,
            "high": prices_high_volume + 1,
            "low": prices_high_volume - 1,
            "close": prices_high_volume,
            "volume": [100000] * 30,
        },
        index=dates,
    )

    low_volume = pd.DataFrame(
        {
            "open": prices_low_volume,
            "high": prices_low_volume + 1,
            "low": prices_low_volume - 1,
            "close": prices_low_volume,
            "volume": [1000] * 30,
        },
        index=dates,
    )

    store.add_ohlcv("HIGH_VOL", high_volume)
    store.add_ohlcv("LOW_VOL", low_volume)

    factor = PriceImpactFactor(lookback=20)
    view = store.get_view(datetime(2024, 1, 30))
    scores = factor.compute(view)

    assert scores["HIGH_VOL"] > scores["LOW_VOL"]


def test_market_depth_proxy_basic():
    store = create_test_store()
    factor = MarketDepthProxyFactor(lookback=20)

    view = store.get_view(datetime(2024, 1, 30))
    scores = factor.compute(view)

    assert "STABLE" in scores or "VOLATILE" in scores


def test_insufficient_data():
    store = DataStore()
    dates = pd.date_range(start=datetime(2024, 1, 1), periods=10, freq="D")

    data = pd.DataFrame(
        {
            "open": [100] * 10,
            "high": [101] * 10,
            "low": [99] * 10,
            "close": [100] * 10,
            "volume": [1000] * 10,
        },
        index=dates,
    )

    store.add_ohlcv("A", data)

    factor = VolumeStabilityFactor(lookback=20)
    view = store.get_view(datetime(2024, 1, 10))
    scores = factor.compute(view)

    assert len(scores) == 0
