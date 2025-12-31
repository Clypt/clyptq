"""
Tests for mean reversion factors.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from clyptq.data.stores.store import DataStore
from clyptq.trading.factors.library.mean_reversion import (
    BollingerFactor,
    PercentileFactor,
    ZScoreFactor,
)


@pytest.fixture
def sample_data():
    """Create sample price data for testing."""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

    store = DataStore()

    for symbol in symbols:
        # Generate trending then mean-reverting price
        trend = np.linspace(100, 150, 100)
        noise = np.random.randn(100) * 5
        prices = trend + noise

        df = pd.DataFrame(
            {
                "open": prices,
                "high": prices * 1.02,
                "low": prices * 0.98,
                "close": prices,
                "volume": np.ones(100) * 1000,
            },
            index=dates,
        )

        store.add_ohlcv(symbol, df)

    return store


def test_bollinger_factor_basic(sample_data):
    """Test BollingerFactor basic functionality."""
    factor = BollingerFactor(lookback=20, num_std=2.0)

    # Get scores at specific timestamp
    timestamp = datetime(2024, 2, 1)
    view = sample_data.get_view(timestamp)
    scores = factor.compute(view)

    # Should return scores for all symbols
    assert len(scores) == 3
    assert "BTC/USDT" in scores
    assert "ETH/USDT" in scores
    assert "SOL/USDT" in scores

    # Scores should be in valid range
    for symbol, score in scores.items():
        assert -1.5 <= score <= 1.5  # Allow slight overflow beyond [-1, 1]


def test_bollinger_overbought_oversold():
    """Test BollingerFactor detects overbought/oversold conditions."""
    factor = BollingerFactor(lookback=20, num_std=2.0)

    dates = pd.date_range(start="2024-01-01", periods=50, freq="D")
    prices = np.ones(50) * 100

    # Make last price very high (overbought)
    prices[-1] = 100 + 3 * 10  # Mean + 3*std

    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices * 1.02,
            "low": prices * 0.98,
            "close": prices,
            "volume": np.ones(50) * 1000,
        },
        index=dates,
    )

    store = DataStore()
    store.add_ohlcv("BTC/USDT", df)
    view = store.get_view(dates[-1])
    scores = factor.compute(view)

    # Should give negative score (sell signal) when overbought
    assert scores["BTC/USDT"] < 0


def test_zscore_factor_basic(sample_data):
    """Test ZScoreFactor basic functionality."""
    factor = ZScoreFactor(lookback=20)

    timestamp = datetime(2024, 2, 1)
    view = sample_data.get_view(timestamp)
    scores = factor.compute(view)

    assert len(scores) == 3
    # Z-scores should be clipped to [-3, 3]
    for score in scores.values():
        assert -3.0 <= score <= 3.0


def test_zscore_mean_reversion():
    """Test ZScoreFactor mean reversion signal."""
    factor = ZScoreFactor(lookback=20)

    dates = pd.date_range(start="2024-01-01", periods=50, freq="D")
    prices = np.ones(50) * 100

    # Make last price deviate significantly from mean
    prices[-1] = 100 + 2 * 10  # Mean + 2*std

    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices * 1.02,
            "low": prices * 0.98,
            "close": prices,
            "volume": np.ones(50) * 1000,
        },
        index=dates,
    )

    store = DataStore()
    store.add_ohlcv("BTC/USDT", df)
    view = store.get_view(dates[-1])
    scores = factor.compute(view)

    assert scores["BTC/USDT"] < 0  # Negative score for high price


def test_percentile_factor_basic(sample_data):
    """Test PercentileFactor basic functionality."""
    factor = PercentileFactor(lookback=20)

    timestamp = datetime(2024, 2, 1)
    view = sample_data.get_view(timestamp)
    scores = factor.compute(view)

    assert len(scores) == 3
    # Percentile scores should be in [-1, 1]
    for score in scores.values():
        assert -1.0 <= score <= 1.0


def test_percentile_extreme_values():
    """Test PercentileFactor at extreme percentiles."""
    factor = PercentileFactor(lookback=20)

    dates = pd.date_range(start="2024-01-01", periods=50, freq="D")
    prices = np.ones(50) * 100

    # Make current price the highest in period
    prices[-1] = 150

    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices * 1.02,
            "low": prices * 0.98,
            "close": prices,
            "volume": np.ones(50) * 1000,
        },
        index=dates,
    )

    store = DataStore()
    store.add_ohlcv("BTC/USDT", df)
    view = store.get_view(dates[-1])
    scores = factor.compute(view)

    # Should be near -1 (overbought, sell signal)
    assert scores["BTC/USDT"] < -0.5


def test_bollinger_zero_volatility():
    """Test BollingerFactor with zero volatility."""
    factor = BollingerFactor(lookback=20)

    dates = pd.date_range(start="2024-01-01", periods=50, freq="D")
    prices = np.ones(50) * 100

    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": np.ones(50) * 1000,
        },
        index=dates,
    )

    store = DataStore()
    store.add_ohlcv("BTC/USDT", df)
    view = store.get_view(dates[-1])
    scores = factor.compute(view)

    # Should return 0 when no volatility
    assert scores["BTC/USDT"] == 0.0


def test_zscore_zero_volatility():
    """Test ZScoreFactor with zero volatility."""
    factor = ZScoreFactor(lookback=20)

    dates = pd.date_range(start="2024-01-01", periods=50, freq="D")
    prices = np.ones(50) * 100

    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": np.ones(50) * 1000,
        },
        index=dates,
    )

    store = DataStore()
    store.add_ohlcv("BTC/USDT", df)
    view = store.get_view(dates[-1])
    scores = factor.compute(view)

    # Should return 0 when no volatility
    assert scores["BTC/USDT"] == 0.0


def test_insufficient_data(sample_data):
    """Test factors handle insufficient data gracefully."""
    factor = BollingerFactor(lookback=200)  # More than available

    timestamp = datetime(2024, 1, 15)  # Early timestamp
    view = sample_data.get_view(timestamp)
    scores = factor.compute(view)

    # Should skip symbols with insufficient data
    assert len(scores) == 0 or all(s == 0 for s in scores.values())
