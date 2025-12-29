"""
Strategy base class and interface.

Defines the interface that all trading strategies must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from clyptq.factors.base import Factor
from clyptq.portfolio.construction import PortfolioConstructor
from clyptq.types import Constraints


class Strategy(ABC):
    """
    Abstract base class for trading strategies.

    Strategies define:
    1. Which alpha factors to use
    2. How to combine them
    3. How to construct portfolios from signals
    4. Rebalancing schedule
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialize strategy.

        Args:
            name: Strategy name (defaults to class name)
        """
        self.name = name or self.__class__.__name__

    @abstractmethod
    def factors(self) -> List[Factor]:
        """
        Get list of alpha factors used by this strategy.

        Returns:
            List of Factor instances
        """
        pass

    @abstractmethod
    def portfolio_constructor(self) -> PortfolioConstructor:
        """
        Get portfolio constructor for this strategy.

        Returns:
            PortfolioConstructor instance
        """
        pass

    @abstractmethod
    def constraints(self) -> Constraints:
        """
        Get portfolio constraints for this strategy.

        Returns:
            Constraints instance
        """
        pass

    def schedule(self) -> str:
        """
        Get rebalancing schedule.

        Returns:
            Schedule string: "daily", "weekly", or "monthly"
        """
        return "daily"

    def universe(self) -> Optional[List[str]]:
        """
        Get trading universe (list of symbols).

        If None, uses all available symbols in data store.

        Returns:
            List of symbols or None for all
        """
        return None

    def warmup_periods(self) -> int:
        """
        Get number of warmup periods required before trading.

        This allows factors to accumulate enough historical data.

        Returns:
            Number of periods to skip at start
        """
        # Default: 100 periods for warmup
        return 100

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name='{self.name}')"


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
