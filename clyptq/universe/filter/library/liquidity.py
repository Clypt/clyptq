"""Liquidity filter using operator-based computation."""

from typing import Optional

from clyptq import operator
from clyptq.universe.filter.base import BaseFilter


class LiquidityFilter(BaseFilter):
    """Filter by average dollar volume (liquidity).

    Keeps symbols with sufficient trading activity.

    Example:
        ```python
        filter = LiquidityFilter(min_dollar_volume=1_000_000, lookback=20)
        ```
    """

    def __init__(
        self,
        min_dollar_volume: float = 1_000_000,
        lookback: int = 20,
        name: Optional[str] = None,
    ):
        """Initialize liquidity filter.

        Args:
            min_dollar_volume: Minimum average daily dollar volume
            lookback: Days for averaging
            name: Optional identifier
        """
        super().__init__(name or "LiquidityFilter")
        self.min_dollar_volume = min_dollar_volume
        self.lookback = lookback

    def compute(self, data):
        """Compute liquidity filter mask.

        Args:
            data: DataProvider with market data

        Returns:
            Boolean mask (T x N) - True = passes filter
        """
        close = data["close"]
        volume = data["volume"]

        # Dollar volume = close * volume
        dollar_volume = operator.mul(close, volume)

        # Rolling average
        avg_dv = operator.ts_mean(dollar_volume, self.lookback)

        # Compare with threshold
        mask = operator.ge(avg_dv, self.min_dollar_volume)

        return mask

    def __repr__(self) -> str:
        return f"LiquidityFilter(min_dv={self.min_dollar_volume:.0e}, lookback={self.lookback})"
