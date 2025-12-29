"""
Mean reversion alpha factors.

Measures price deviations from statistical norms for mean reversion signals.
"""

import numpy as np

from clyptq.data.store import DataView
from clyptq.factors.base import Factor
from typing import Dict


class BollingerFactor(Factor):
    """
    Bollinger Bands mean reversion factor.

    Measures price position relative to Bollinger Bands.
    Negative score when price above upper band (overbought).
    Positive score when price below lower band (oversold).
    """

    def __init__(self, lookback: int = 20, num_std: float = 2.0, name: str = "Bollinger"):
        """
        Initialize Bollinger factor.

        Args:
            lookback: Period for moving average and std
            num_std: Number of standard deviations for bands
            name: Factor name
        """
        super().__init__(name)
        self.lookback = lookback
        self.num_std = num_std

    def compute(self, data: DataView) -> Dict[str, float]:
        """
        Compute Bollinger band mean reversion scores.

        Score = (middle - price) / (upper - middle)
        Positive when oversold, negative when overbought.

        Args:
            data: DataView at current timestamp

        Returns:
            Dictionary of {symbol: bollinger_score}
        """
        scores = {}

        for symbol in data.symbols:
            try:
                prices = data.close(symbol, self.lookback)

                current_price = prices[-1]
                middle = np.mean(prices)
                std = np.std(prices)

                upper = middle + self.num_std * std
                lower = middle - self.num_std * std

                bandwidth = upper - middle

                if bandwidth < 1e-10:
                    scores[symbol] = 0.0
                else:
                    # Normalize deviation by bandwidth
                    score = (middle - current_price) / bandwidth
                    # Clip to [-1, 1]
                    scores[symbol] = np.clip(score, -1.0, 1.0)

            except (KeyError, ValueError):
                continue

        return scores


class ZScoreFactor(Factor):
    """
    Z-Score mean reversion factor.

    Measures how many standard deviations price is from its mean.
    """

    def __init__(self, lookback: int = 20, name: str = "ZScore"):
        """
        Initialize Z-Score factor.

        Args:
            lookback: Period for mean and std calculation
            name: Factor name
        """
        super().__init__(name)
        self.lookback = lookback

    def compute(self, data: DataView) -> Dict[str, float]:
        """
        Compute Z-Score mean reversion scores.

        Score = -(price - mean) / std
        Negative sign so high price gives negative score (sell signal).

        Args:
            data: DataView at current timestamp

        Returns:
            Dictionary of {symbol: zscore}
        """
        scores = {}

        for symbol in data.symbols:
            try:
                prices = data.close(symbol, self.lookback)

                current_price = prices[-1]
                mean = np.mean(prices)
                std = np.std(prices)

                if std < 1e-10:
                    scores[symbol] = 0.0
                else:
                    zscore = -(current_price - mean) / std
                    # Clip to [-3, 3] range
                    scores[symbol] = np.clip(zscore, -3.0, 3.0)

            except (KeyError, ValueError):
                continue

        return scores


class PercentileFactor(Factor):
    """
    Percentile-based mean reversion factor.

    Measures current price percentile in historical distribution.
    High percentile (near 100) means overbought, low percentile means oversold.
    """

    def __init__(self, lookback: int = 20, name: str = "Percentile"):
        """
        Initialize Percentile factor.

        Args:
            lookback: Period for percentile calculation
            name: Factor name
        """
        super().__init__(name)
        self.lookback = lookback

    def compute(self, data: DataView) -> Dict[str, float]:
        """
        Compute percentile mean reversion scores.

        Score = -(percentile - 50) / 50
        Normalized to [-1, 1] range.

        Args:
            data: DataView at current timestamp

        Returns:
            Dictionary of {symbol: percentile_score}
        """
        scores = {}

        for symbol in data.symbols:
            try:
                prices = data.close(symbol, self.lookback)
                current_price = prices[-1]

                # Calculate percentile rank
                percentile = (prices < current_price).sum() / len(prices) * 100

                # Normalize to [-1, 1], negative when overbought
                score = -(percentile - 50.0) / 50.0

                scores[symbol] = score

            except (KeyError, ValueError):
                continue

        return scores
