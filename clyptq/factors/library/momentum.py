"""
Momentum-based alpha factors.

Measures price trends and momentum signals.
"""

import numpy as np

from clyptq.data.store import DataView
from clyptq.factors.base import Factor
from typing import Dict


class MomentumFactor(Factor):
    """
    Simple momentum factor based on price returns.

    Computes returns over a lookback period as the alpha signal.
    """

    def __init__(self, lookback: int = 20, name: str = "Momentum"):
        """
        Initialize momentum factor.

        Args:
            lookback: Lookback period in bars
            name: Factor name
        """
        super().__init__(name)
        self.lookback = lookback

    def compute(self, data: DataView) -> Dict[str, float]:
        """
        Compute momentum scores.

        Score = (current_price - past_price) / past_price

        Args:
            data: DataView at current timestamp

        Returns:
            Dictionary of {symbol: momentum_score}
        """
        scores = {}

        for symbol in data.symbols:
            try:
                # Get price history (need lookback + 1 for returns)
                prices = data.close(symbol, self.lookback + 1)

                # Compute momentum as total return
                momentum = (prices[-1] - prices[0]) / prices[0]

                scores[symbol] = momentum

            except (KeyError, ValueError):
                # Skip symbols with insufficient data
                continue

        return scores


class RSIFactor(Factor):
    """
    RSI (Relative Strength Index) momentum factor.

    Measures momentum using RSI oscillator (0-100).
    """

    def __init__(self, lookback: int = 14, name: str = "RSI"):
        """
        Initialize RSI factor.

        Args:
            lookback: RSI period
            name: Factor name
        """
        super().__init__(name)
        self.lookback = lookback

    def compute(self, data: DataView) -> Dict[str, float]:
        """
        Compute RSI scores.

        RSI = 100 - 100 / (1 + RS)
        where RS = average_gain / average_loss

        Args:
            data: DataView at current timestamp

        Returns:
            Dictionary of {symbol: rsi_score}
            Scores normalized to [-1, 1] range
        """
        scores = {}

        for symbol in data.symbols:
            try:
                # Get price history (need extra for returns)
                prices = data.close(symbol, self.lookback + 2)

                # Calculate price changes
                deltas = np.diff(prices)

                # Separate gains and losses
                gains = np.where(deltas > 0, deltas, 0)
                losses = np.where(deltas < 0, -deltas, 0)

                # Calculate average gain and loss
                avg_gain = np.mean(gains)
                avg_loss = np.mean(losses)

                # Avoid division by zero
                if avg_loss < 1e-10:
                    rsi = 100.0
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100.0 - (100.0 / (1.0 + rs))

                # Normalize to [-1, 1]  range
                # RSI > 70 is overbought (positive), RSI < 30 is oversold (negative)
                normalized_rsi = (rsi - 50.0) / 50.0

                scores[symbol] = normalized_rsi

            except (KeyError, ValueError):
                continue

        return scores


class TrendStrengthFactor(Factor):
    """
    Trend strength factor using linear regression slope.

    Measures strength and direction of price trend.
    """

    def __init__(self, lookback: int = 20, name: str = "TrendStrength"):
        """
        Initialize trend strength factor.

        Args:
            lookback: Lookback period for trend
            name: Factor name
        """
        super().__init__(name)
        self.lookback = lookback

    def compute(self, data: DataView) -> Dict[str, float]:
        """
        Compute trend strength using linear regression slope.

        Args:
            data: DataView at current timestamp

        Returns:
            Dictionary of {symbol: trend_strength}
        """
        scores = {}

        for symbol in data.symbols:
            try:
                # Get price history
                prices = data.close(symbol, self.lookback)
                log_prices = np.log(prices)

                # Linear regression
                x = np.arange(len(log_prices))
                slope = np.polyfit(x, log_prices, 1)[0]

                # Annualize slope (approximate)
                annualized_slope = slope * 252  # Assuming daily data

                scores[symbol] = annualized_slope

            except (KeyError, ValueError):
                continue

        return scores
