"""
Volatility-based alpha factors.

Measures price volatility and risk signals.
"""

import numpy as np

from clypt.data.store import DataView
from clypt.factors.base import Factor
from typing import Dict


class VolatilityFactor(Factor):
    """
    Historical volatility factor.

    Computes realized volatility over a lookback period.
    """

    def __init__(self, lookback: int = 20, annualize: bool = True, name: str = "Volatility"):
        """
        Initialize volatility factor.

        Args:
            lookback: Lookback period in bars
            annualize: If True, annualize volatility (assumes daily data)
            name: Factor name
        """
        super().__init__(name)
        self.lookback = lookback
        self.annualize = annualize

    def compute(self, data: DataView) -> Dict[str, float]:
        """
        Compute volatility scores.

        Volatility = std(returns) * sqrt(252) if annualize

        Args:
            data: DataView at current timestamp

        Returns:
            Dictionary of {symbol: volatility}
        """
        scores = {}

        for symbol in data.symbols:
            try:
                # Get returns
                returns = data.returns(symbol, self.lookback)

                # Calculate volatility
                volatility = np.std(returns)

                # Annualize if requested
                if self.annualize:
                    volatility *= np.sqrt(252)

                scores[symbol] = volatility

            except (KeyError, ValueError):
                continue

        return scores


class ATRFactor(Factor):
    """
    Average True Range (ATR) volatility factor.

    Measures volatility using average true range.
    """

    def __init__(self, lookback: int = 14, name: str = "ATR"):
        """
        Initialize ATR factor.

        Args:
            lookback: ATR period
            name: Factor name
        """
        super().__init__(name)
        self.lookback = lookback

    def compute(self, data: DataView) -> Dict[str, float]:
        """
        Compute ATR scores.

        TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
        ATR = average(TR over lookback period)

        Args:
            data: DataView at current timestamp

        Returns:
            Dictionary of {symbol: atr}
        """
        scores = {}

        for symbol in data.symbols:
            try:
                # Get OHLCV data
                ohlcv = data.ohlcv(symbol, self.lookback + 1)

                # Calculate true range
                high_low = ohlcv["high"] - ohlcv["low"]
                high_close = np.abs(ohlcv["high"] - ohlcv["close"].shift(1))
                low_close = np.abs(ohlcv["low"] - ohlcv["close"].shift(1))

                true_range = np.maximum(high_low, np.maximum(high_close, low_close))

                # Average true range (skip first NaN from shift)
                atr = true_range[1:].mean()

                # Normalize by price
                current_price = ohlcv["close"].iloc[-1]
                normalized_atr = atr / current_price if current_price > 0 else 0.0

                scores[symbol] = normalized_atr

            except (KeyError, ValueError):
                continue

        return scores


class RangeVolatilityFactor(Factor):
    """
    Range-based volatility factor.

    Measures volatility using high-low price ranges.
    """

    def __init__(self, lookback: int = 20, name: str = "RangeVolatility"):
        """
        Initialize range volatility factor.

        Args:
            lookback: Lookback period
            name: Factor name
        """
        super().__init__(name)
        self.lookback = lookback

    def compute(self, data: DataView) -> Dict[str, float]:
        """
        Compute range volatility.

        Range = (high - low) / close

        Args:
            data: DataView at current timestamp

        Returns:
            Dictionary of {symbol: range_volatility}
        """
        scores = {}

        for symbol in data.symbols:
            try:
                # Get OHLCV data
                ohlcv = data.ohlcv(symbol, self.lookback)

                # Calculate normalized ranges
                ranges = (ohlcv["high"] - ohlcv["low"]) / ohlcv["close"]

                # Average range
                avg_range = ranges.mean()

                scores[symbol] = avg_range

            except (KeyError, ValueError):
                continue

        return scores
