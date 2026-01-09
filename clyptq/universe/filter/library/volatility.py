"""Volatility filter using operator-based computation."""

from typing import Optional

from clyptq import operator
from clyptq.universe.filter.base import BaseFilter


class VolatilityFilter(BaseFilter):
    """Filter by volatility range.

    Excludes extremely volatile or dead symbols.

    Example:
        ```python
        filter = VolatilityFilter(min_vol=0.01, max_vol=0.5, lookback=20)
        ```
    """

    def __init__(
        self,
        min_vol: float = 0.001,
        max_vol: float = 1.0,
        lookback: int = 20,
        name: Optional[str] = None,
    ):
        """Initialize volatility filter.

        Args:
            min_vol: Minimum annualized volatility
            max_vol: Maximum annualized volatility
            lookback: Days for volatility calculation
            name: Optional identifier
        """
        super().__init__(name or "VolatilityFilter")
        self.min_vol = min_vol
        self.max_vol = max_vol
        self.lookback = lookback

    def compute(self, data):
        """Compute volatility filter mask.

        Args:
            data: DataProvider with market data

        Returns:
            Boolean mask (T x N) - True = passes filter
        """
        close = data["close"]

        # Calculate returns
        returns = operator.ts_returns(close)

        # Rolling volatility (annualized)
        vol = operator.ts_std(returns, self.lookback)
        annualized_vol = operator.mul(vol, 252 ** 0.5)  # Annualize

        # Range check
        min_mask = operator.ge(annualized_vol, self.min_vol)
        max_mask = operator.le(annualized_vol, self.max_vol)
        mask = operator.logical_and(min_mask, max_mask)

        return mask

    def __repr__(self) -> str:
        return f"VolatilityFilter(min={self.min_vol}, max={self.max_vol})"
