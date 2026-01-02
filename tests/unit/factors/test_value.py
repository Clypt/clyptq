import numpy as np
import pandas as pd
from datetime import datetime

from clyptq.data.stores.store import DataStore
from clyptq.trading.factors.library.value import (
    ImpliedBasisFactor,
    PriceEfficiencyFactor,
    RealizedSpreadFactor,
)


def create_test_store():
    store = DataStore()
    dates = pd.date_range(start=datetime(2024, 1, 1), periods=30, freq="D")

    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(30) * 0.5)

    tight_spread = pd.DataFrame(
        {
            "open": prices,
            "high": prices + 0.1,
            "low": prices - 0.1,
            "close": prices,
            "volume": [10000] * 30,
        },
        index=dates,
    )

    wide_spread = pd.DataFrame(
        {
            "open": prices,
            "high": prices + 5,
            "low": prices - 5,
            "close": prices,
            "volume": [10000] * 30,
        },
        index=dates,
    )

    store.add_ohlcv("TIGHT", tight_spread)
    store.add_ohlcv("WIDE", wide_spread)

    return store


def test_realized_spread_basic():
    store = create_test_store()
    factor = RealizedSpreadFactor(lookback=20)

    view = store.get_view(datetime(2024, 1, 30))
    scores = factor.compute(view)

    assert "TIGHT" in scores
    assert "WIDE" in scores


def test_realized_spread_prefers_tight():
    store = create_test_store()
    factor = RealizedSpreadFactor(lookback=20)

    view = store.get_view(datetime(2024, 1, 30))
    scores = factor.compute(view)

    assert scores["TIGHT"] > scores["WIDE"]


def test_price_efficiency_basic():
    store = create_test_store()
    factor = PriceEfficiencyFactor(lookback=20)

    view = store.get_view(datetime(2024, 1, 30))
    scores = factor.compute(view)

    assert "TIGHT" in scores
    assert "WIDE" in scores


def test_implied_basis_basic():
    store = DataStore()
    dates = pd.date_range(start=datetime(2024, 1, 1), periods=30, freq="D")

    np.random.seed(42)
    uptrend_prices = 100 + np.arange(30) * 0.5
    downtrend_prices = 100 - np.arange(30) * 0.5

    uptrend = pd.DataFrame(
        {
            "open": uptrend_prices,
            "high": uptrend_prices + 1,
            "low": uptrend_prices - 1,
            "close": uptrend_prices,
            "volume": [1000] * 30,
        },
        index=dates,
    )

    downtrend = pd.DataFrame(
        {
            "open": downtrend_prices,
            "high": downtrend_prices + 1,
            "low": downtrend_prices - 1,
            "close": downtrend_prices,
            "volume": [1000] * 30,
        },
        index=dates,
    )

    store.add_ohlcv("UP", uptrend)
    store.add_ohlcv("DOWN", downtrend)

    factor = ImpliedBasisFactor(lookback=20)
    view = store.get_view(datetime(2024, 1, 30))
    scores = factor.compute(view)

    assert "UP" in scores
    assert "DOWN" in scores


def test_implied_basis_direction():
    store = DataStore()
    dates = pd.date_range(start=datetime(2024, 1, 1), periods=30, freq="D")

    uptrend_prices = 100 + np.arange(30) * 0.5
    downtrend_prices = 100 - np.arange(30) * 0.5

    uptrend = pd.DataFrame(
        {
            "open": uptrend_prices,
            "high": uptrend_prices + 1,
            "low": uptrend_prices - 1,
            "close": uptrend_prices,
            "volume": [1000] * 30,
        },
        index=dates,
    )

    downtrend = pd.DataFrame(
        {
            "open": downtrend_prices,
            "high": downtrend_prices + 1,
            "low": downtrend_prices - 1,
            "close": downtrend_prices,
            "volume": [1000] * 30,
        },
        index=dates,
    )

    store.add_ohlcv("UP", uptrend)
    store.add_ohlcv("DOWN", downtrend)

    factor = ImpliedBasisFactor(lookback=20)
    view = store.get_view(datetime(2024, 1, 30))
    scores = factor.compute(view)

    assert scores["UP"] > scores["DOWN"]


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

    factor = RealizedSpreadFactor(lookback=20)
    view = store.get_view(datetime(2024, 1, 10))
    scores = factor.compute(view)

    assert len(scores) == 0
