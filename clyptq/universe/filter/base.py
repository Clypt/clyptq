"""Base class for universe filters.

All filters MUST use clyptq operators only - no pandas/numpy direct usage.
This ensures consistency between backtest and live, and enables future Rust migration.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import pandas as pd

if TYPE_CHECKING:
    from clyptq.data.provider import DataProvider


class BaseFilter(ABC):
    """Abstract base class for universe filters.

    Filters return a boolean mask (DataFrame) indicating which symbols pass.
    All filter implementations MUST use operators only - no pandas/numpy direct usage.

    Example:
        ```python
        class MyFilter(BaseFilter):
            def __init__(self, threshold: float):
                super().__init__(name="MyFilter")
                self.threshold = threshold

            def compute(self, data: DataProvider) -> pd.DataFrame:
                close = data["close"]
                return operator.ge(close, self.threshold)
        ```
    """

    def __init__(self, name: Optional[str] = None):
        """Initialize filter.

        Args:
            name: Filter name (defaults to class name)
        """
        self.name = name or self.__class__.__name__

    @abstractmethod
    def compute(self, data: "DataProvider") -> pd.DataFrame:
        """Compute filter mask.

        Args:
            data: DataProvider with market data

        Returns:
            Boolean DataFrame (T x N) - True = passes filter
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __and__(self, other: "BaseFilter") -> "CompositeFilter":
        """Combine filters with AND logic using & operator."""
        from clyptq.universe.filter.library.composite import CompositeFilter
        return CompositeFilter([self, other], logic="and")

    def __or__(self, other: "BaseFilter") -> "CompositeFilter":
        """Combine filters with OR logic using | operator."""
        from clyptq.universe.filter.library.composite import CompositeFilter
        return CompositeFilter([self, other], logic="or")


# Legacy alias for backward compatibility
UniverseFilter = BaseFilter
