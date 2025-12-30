import numpy as np
import pandas as pd
from datetime import datetime

from clyptq.data.stores.store import DataStore
from clyptq.factors.library.size import DollarVolumeSizeFactor


def create_test_store():
    store = DataStore()
    dates = pd.date_range(start=datetime(2024, 1, 1), periods=30, freq="D")

    data_large = pd.DataFrame(
        {
            "open": [1000] * 30,
            "high": [1010] * 30,
            "low": [990] * 30,
            "close": [1000] * 30,
            "volume": [100000] * 30,
        },
        index=dates,
    )

    data_small = pd.DataFrame(
        {
            "open": [100] * 30,
            "high": [101] * 30,
            "low": [99] * 30,
            "close": [100] * 30,
            "volume": [1000] * 30,
        },
        index=dates,
    )

    store.add_ohlcv("LARGE", data_large)
    store.add_ohlcv("SMALL", data_small)

    return store


def test_dollar_volume_size_basic():
    store = create_test_store()
    factor = DollarVolumeSizeFactor(lookback=20)

    view = store.get_view(datetime(2024, 1, 30))
    scores = factor.compute(view)

    assert "LARGE" in scores
    assert "SMALL" in scores


def test_dollar_volume_size_ordering():
    store = create_test_store()
    factor = DollarVolumeSizeFactor(lookback=20)

    view = store.get_view(datetime(2024, 1, 30))
    scores = factor.compute(view)

    assert scores["LARGE"] > scores["SMALL"]


def test_log_transformation():
    store = create_test_store()
    factor = DollarVolumeSizeFactor(lookback=20)

    view = store.get_view(datetime(2024, 1, 30))
    scores = factor.compute(view)

    assert scores["LARGE"] > 0
    assert scores["SMALL"] > 0
    assert scores["LARGE"] < scores["SMALL"] * 10


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

    factor = DollarVolumeSizeFactor(lookback=20)
    view = store.get_view(datetime(2024, 1, 10))
    scores = factor.compute(view)

    assert len(scores) == 0
