"""Simple strategy implementation."""

from typing import List, Optional

from clyptq.core.base import Factor, PortfolioConstructor, Strategy
from clyptq.core.types import Constraints


class SimpleStrategy(Strategy):
    """
    Simple strategy implementation for quick prototyping.

    Allows creating a strategy by passing components directly.
    """

    def __init__(
        self,
        factors_list: List[Factor],
        constructor: PortfolioConstructor,
        constraints_obj: Constraints,
        schedule_str: str = "daily",
        warmup: int = 100,
        name: Optional[str] = None,
    ):
        """
        Initialize simple strategy.

        Args:
            factors_list: List of factors to use
            constructor: Portfolio constructor
            constraints_obj: Portfolio constraints
            schedule_str: Rebalancing schedule
            warmup: Warmup periods
            name: Strategy name
        """
        super().__init__(name)
        self._factors = factors_list
        self._constructor = constructor
        self._constraints = constraints_obj
        self._schedule = schedule_str
        self._warmup = warmup

    def factors(self) -> List[Factor]:
        """Get factors."""
        return self._factors

    def portfolio_constructor(self) -> PortfolioConstructor:
        """Get portfolio constructor."""
        return self._constructor

    def constraints(self) -> Constraints:
        """Get constraints."""
        return self._constraints

    def schedule(self) -> str:
        """Get schedule."""
        return self._schedule

    def warmup_periods(self) -> int:
        """Get warmup periods."""
        return self._warmup
