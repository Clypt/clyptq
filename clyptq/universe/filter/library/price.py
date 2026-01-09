"""Price filter using operator-based computation."""

from typing import Optional

from clyptq import operator
from clyptq.universe.filter.base import BaseFilter


class PriceFilter(BaseFilter):
    """Filter by price range.

    Excludes penny stocks or very expensive stocks.

    Example:
        ```python
        filter = PriceFilter(min_price=5.0, max_price=10000.0)
        ```
    """

    def __init__(
        self,
        min_price: float = 1.0,
        max_price: Optional[float] = None,
        name: Optional[str] = None,
    ):
        """Initialize price filter.

        Args:
            min_price: Minimum price threshold
            max_price: Maximum price threshold (None = no limit)
            name: Optional identifier
        """
        super().__init__(name or "PriceFilter")
        self.min_price = min_price
        self.max_price = max_price

    def compute(self, data):
        """Compute price filter mask.

        Args:
            data: DataProvider with market data

        Returns:
            Boolean mask (T x N) - True = passes filter
        """
        close = data["close"]

        # Min price check
        mask = operator.ge(close, self.min_price)

        # Max price check (if specified)
        if self.max_price is not None:
            max_mask = operator.le(close, self.max_price)
            mask = operator.logical_and(mask, max_mask)

        return mask

    def __repr__(self) -> str:
        max_str = f", max={self.max_price}" if self.max_price else ""
        return f"PriceFilter(min={self.min_price}{max_str})"
