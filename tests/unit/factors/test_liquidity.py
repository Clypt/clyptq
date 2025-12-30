import numpy as np
import pandas as pd
from datetime import datetime

from clyptq.data.stores.store import DataStore
from clyptq.factors.library.liquidity import (
    AmihudFactor,
    EffectiveSpreadFactor,
    VolatilityOfVolatilityFactor,
)


def create_test_store():
    store = DataStore()
    dates = pd.date_range(start=datetime(2024, 1, 1), periods=30, freq="D")

    np.random.seed(42)
    liquid_prices = 100 + np.cumsum(np.random.randn(30) * 0.5)
    illiquid_prices = 100 + np.cumsum(np.random.randn(30) * 2)

    data_liquid = pd.DataFrame(
        {
            "open": liquid_prices,
            "high": liquid_prices + 0.5,
            "low": liquid_prices - 0.5,
            "close": liquid_prices,
            "volume": [10000] * 30,
        },
        index=dates,
    )

    data_illiquid = pd.DataFrame(
        {
            "open": illiquid_prices,
            "high": illiquid_prices + 5,
            "low": illiquid_prices - 5,
            "close": illiquid_prices,
            "volume": [1000] * 30,
        },
        index=dates,
    )

    store.add_ohlcv("LIQUID", data_liquid)
    store.add_ohlcv("ILLIQUID", data_illiquid)

    return store


def test_amihud_factor_basic():
    store = create_test_store()
    factor = AmihudFactor(lookback=20)

    view = store.get_view(datetime(2024, 1, 30))
    scores = factor.compute(view)

    assert "LIQUID" in scores
    assert "ILLIQUID" in scores


def test_amihud_prefers_liquid():
    store = create_test_store()
    factor = AmihudFactor(lookback=20)

    view = store.get_view(datetime(2024, 1, 30))
    scores = factor.compute(view)

    assert scores["LIQUID"] > scores["ILLIQUID"]


def test_effective_spread_basic():
    store = create_test_store()
    factor = EffectiveSpreadFactor(lookback=20)

    view = store.get_view(datetime(2024, 1, 30))
    scores = factor.compute(view)

    assert "LIQUID" in scores
    assert "ILLIQUID" in scores


def test_effective_spread_prefers_narrow():
    store = create_test_store()
    factor = EffectiveSpreadFactor(lookback=20)

    view = store.get_view(datetime(2024, 1, 30))
    scores = factor.compute(view)

    assert scores["LIQUID"] > scores["ILLIQUID"]


def test_volatility_of_volatility_basic():
    store = DataStore()
    dates = pd.date_range(start=datetime(2024, 1, 1), periods=50, freq="D")

    np.random.seed(42)
    prices_stable = 100 + np.random.randn(50) * 1
    prices_volatile = 100 + np.random.randn(50) * 5

    data_stable = pd.DataFrame(
        {
            "open": prices_stable,
            "high": prices_stable + 1,
            "low": prices_stable - 1,
            "close": prices_stable,
            "volume": [1000] * 50,
        },
        index=dates,
    )

    data_volatile = pd.DataFrame(
        {
            "open": prices_volatile,
            "high": prices_volatile + 1,
            "low": prices_volatile - 1,
            "close": prices_volatile,
            "volume": [1000] * 50,
        },
        index=dates,
    )

    store.add_ohlcv("STABLE", data_stable)
    store.add_ohlcv("VOLATILE", data_volatile)

    factor = VolatilityOfVolatilityFactor(lookback=20, vol_window=5)
    view = store.get_view(datetime(2024, 2, 19))
    scores = factor.compute(view)

    assert "STABLE" in scores
    assert "VOLATILE" in scores


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

    factor = AmihudFactor(lookback=20)
    view = store.get_view(datetime(2024, 1, 10))
    scores = factor.compute(view)

    assert len(scores) == 0
