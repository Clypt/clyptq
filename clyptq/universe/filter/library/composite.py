"""Composite filter for combining multiple filters."""

from typing import List, Literal, Optional

from clyptq import operator
from clyptq.universe.filter.base import BaseFilter


class CompositeFilter(BaseFilter):
    """Combine multiple filters with AND/OR logic.

    Example:
        ```python
        # AND logic (default)
        composite = CompositeFilter([
            LiquidityFilter(min_dollar_volume=1_000_000),
            PriceFilter(min_price=5.0),
            VolumeFilter(min_volume=100_000),
        ])

        # OR logic
        composite = CompositeFilter([
            LiquidityFilter(min_dollar_volume=1_000_000),
            VolumeFilter(min_volume=500_000),
        ], logic="or")

        # Using operators
        filter1 = LiquidityFilter(min_dollar_volume=1e6)
        filter2 = PriceFilter(min_price=5.0)
        combined = filter1 & filter2  # AND
        combined = filter1 | filter2  # OR
        ```
    """

    def __init__(
        self,
        filters: List[BaseFilter],
        logic: Literal["and", "or"] = "and",
        name: Optional[str] = None,
    ):
        """Initialize composite filter.

        Args:
            filters: List of filters to combine
            logic: "and" or "or" logic for combining
            name: Optional identifier
        """
        super().__init__(name or "CompositeFilter")
        self._filters = filters
        self._logic = logic

    def compute(self, data):
        """Compute composite filter mask.

        Args:
            data: DataProvider with market data

        Returns:
            Boolean mask (T x N) - True = passes filter
        """
        close = data["close"]

        if self._logic == "and":
            # Start with all True (AND logic)
            mask = operator.notna(close)
            for filter_ in self._filters:
                filter_mask = filter_.compute(data)
                mask = operator.logical_and(mask, filter_mask)
        else:
            # Start with all False (OR logic)
            mask = operator.eq(close, float('inf'))  # All False
            for filter_ in self._filters:
                filter_mask = filter_.compute(data)
                mask = operator.logical_or(mask, filter_mask)

        return mask

    def __repr__(self) -> str:
        return f"CompositeFilter({len(self._filters)} filters, logic={self._logic})"
