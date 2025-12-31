import numpy as np
import pandas as pd
from datetime import datetime

from clyptq.data.stores.store import DataStore
from clyptq.trading.factors.library.volume import (
    DollarVolumeFactor,
    VolumeFactor,
    VolumeRatioFactor,
)


def create_test_store():
    store = DataStore()
    dates = pd.date_range(start=datetime(2024, 1, 1), periods=30, freq="D")

    data_a = pd.DataFrame(
        {
            "open": np.linspace(100, 110, 30),
            "high": np.linspace(101, 111, 30),
            "low": np.linspace(99, 109, 30),
            "close": np.linspace(100, 110, 30),
            "volume": np.linspace(1000, 2000, 30),
        },
        index=dates,
    )

    data_b = pd.DataFrame(
        {
            "open": [100] * 30,
            "high": [101] * 30,
            "low": [99] * 30,
            "close": [100] * 30,
            "volume": [1000] * 30,
        },
        index=dates,
    )

    store.add_ohlcv("A", data_a)
    store.add_ohlcv("B", data_b)

    return store


def test_volume_factor_basic():
    store = create_test_store()
    factor = VolumeFactor(lookback=20)

    view = store.get_view(datetime(2024, 1, 30))
    scores = factor.compute(view)

    assert "A" in scores
    assert "B" in scores
    assert scores["A"] > scores["B"]


def test_volume_factor_increasing_volume():
    store = create_test_store()
    factor = VolumeFactor(lookback=20)

    view = store.get_view(datetime(2024, 1, 30))
    scores = factor.compute(view)

    assert scores["A"] > 1.0
    assert abs(scores["B"] - 1.0) < 0.01


def test_dollar_volume_factor_basic():
    store = create_test_store()
    factor = DollarVolumeFactor(lookback=20)

    view = store.get_view(datetime(2024, 1, 30))
    scores = factor.compute(view)

    assert "A" in scores
    assert "B" in scores
    assert scores["A"] > scores["B"]


def test_volume_ratio_factor_basic():
    store = create_test_store()
    factor = VolumeRatioFactor(short_window=5, long_window=20)

    view = store.get_view(datetime(2024, 1, 30))
    scores = factor.compute(view)

    assert "A" in scores
    assert "B" in scores


def test_volume_ratio_increasing():
    store = create_test_store()
    factor = VolumeRatioFactor(short_window=5, long_window=20)

    view = store.get_view(datetime(2024, 1, 30))
    scores = factor.compute(view)

    assert scores["A"] > scores["B"]
    assert scores["A"] > 1.0


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

    factor = VolumeFactor(lookback=20)
    view = store.get_view(datetime(2024, 1, 10))
    scores = factor.compute(view)

    assert len(scores) == 0
