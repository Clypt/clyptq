"""Data availability filter using operator-based computation."""

from typing import Optional

from clyptq import operator
from clyptq.universe.filter.base import BaseFilter


class DataAvailabilityFilter(BaseFilter):
    """Filter by data availability.

    Keeps symbols with sufficient historical data.

    Example:
        ```python
        filter = DataAvailabilityFilter(min_bars=100)
        ```
    """

    def __init__(
        self,
        min_bars: int = 50,
        name: Optional[str] = None,
    ):
        """Initialize data availability filter.

        Args:
            min_bars: Minimum number of valid bars required
            name: Optional identifier
        """
        super().__init__(name or "DataAvailabilityFilter")
        self.min_bars = min_bars

    def compute(self, data):
        """Compute data availability filter mask.

        Args:
            data: DataProvider with market data

        Returns:
            Boolean mask (T x N) - True = passes filter
        """
        close = data["close"]

        # Convert boolean to float: True=1.0, False=0.0
        # Use condition operator instead of .astype(float)
        valid_mask = operator.notna(close)
        valid_float = operator.condition(valid_mask, 1.0, 0.0)

        # Count valid (non-NaN) values per symbol using rolling window
        valid_count = operator.ts_sum(valid_float, self.min_bars)

        # Check if we have enough data
        mask = operator.ge(valid_count, self.min_bars)

        return mask

    def __repr__(self) -> str:
        return f"DataAvailabilityFilter(min_bars={self.min_bars})"
