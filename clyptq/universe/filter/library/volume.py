"""Volume filter using operator-based computation."""

from typing import Optional

from clyptq import operator
from clyptq.universe.filter.base import BaseFilter


class VolumeFilter(BaseFilter):
    """Filter by minimum trading volume.

    Example:
        ```python
        filter = VolumeFilter(min_volume=100_000, lookback=20)
        ```
    """

    def __init__(
        self,
        min_volume: float = 100_000,
        lookback: int = 20,
        name: Optional[str] = None,
    ):
        """Initialize volume filter.

        Args:
            min_volume: Minimum average daily volume
            lookback: Days for averaging
            name: Optional identifier
        """
        super().__init__(name or "VolumeFilter")
        self.min_volume = min_volume
        self.lookback = lookback

    def compute(self, data):
        """Compute volume filter mask.

        Args:
            data: DataProvider with market data

        Returns:
            Boolean mask (T x N) - True = passes filter
        """
        volume = data["volume"]

        # Rolling average
        avg_volume = operator.ts_mean(volume, self.lookback)

        # Compare with threshold
        mask = operator.ge(avg_volume, self.min_volume)

        return mask

    def __repr__(self) -> str:
        return f"VolumeFilter(min_vol={self.min_volume:.0e}, lookback={self.lookback})"
