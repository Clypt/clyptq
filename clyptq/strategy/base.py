"""
Strategy base class and interface.

Defines the interface that all trading strategies must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from clyptq.factors.base import Factor
from clyptq.portfolio.construction import PortfolioConstructor
from clyptq.core.types import Constraints


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


class StrategyBlender(Strategy):
    """
    Multi-strategy blender for portfolio allocation across strategies.

    Combines multiple strategies with specified allocations, running each
    independently and blending their target weights.
    """

    def __init__(
        self,
        strategies: Dict[str, Strategy],
        allocations: Dict[str, float],
        name: Optional[str] = None,
    ):
        """
        Initialize strategy blender.

        Args:
            strategies: Dict of {strategy_name: Strategy}
            allocations: Dict of {strategy_name: allocation_weight}
            name: Blender name

        Raises:
            ValueError: If strategies and allocations don't match or allocations don't sum to 1.0
        """
        super().__init__(name or "StrategyBlender")

        if set(strategies.keys()) != set(allocations.keys()):
            raise ValueError("Strategies and allocations must have same keys")

        total_alloc = sum(allocations.values())
        if abs(total_alloc - 1.0) > 1e-6:
            raise ValueError(f"Allocations must sum to 1.0, got {total_alloc}")

        self.strategies = strategies
        self.allocations = allocations
        self._tagged_factors: List[Factor] = []
        self._factor_map: Dict[str, str] = {}

        for strategy_name, strategy in strategies.items():
            for factor in strategy.factors():
                tagged_name = f"{strategy_name}_{factor.name}"
                tagged_factor = _TaggedFactor(factor, tagged_name)
                self._tagged_factors.append(tagged_factor)
                self._factor_map[tagged_name] = strategy_name

    def factors(self) -> List[Factor]:
        """Get all tagged factors from all strategies."""
        return self._tagged_factors

    def portfolio_constructor(self) -> PortfolioConstructor:
        """Get blended constructor."""
        from clyptq.portfolio.construction import BlendedConstructor

        return BlendedConstructor(self.strategies, self.allocations, self._factor_map)

    def constraints(self) -> Constraints:
        """Get combined constraints (most restrictive)."""
        all_constraints = [s.constraints() for s in self.strategies.values()]

        return Constraints(
            max_position_size=min(c.max_position_size for c in all_constraints),
            min_position_size=max(c.min_position_size for c in all_constraints),
            max_gross_exposure=min(c.max_gross_exposure for c in all_constraints),
            max_num_positions=min(c.max_num_positions for c in all_constraints),
            allow_short=all(c.allow_short for c in all_constraints),
        )

    def schedule(self) -> str:
        """Get most frequent schedule among strategies."""
        schedules = [s.schedule() for s in self.strategies.values()]
        if "daily" in schedules:
            return "daily"
        elif "weekly" in schedules:
            return "weekly"
        else:
            return "monthly"

    def warmup_periods(self) -> int:
        """Get maximum warmup among strategies."""
        return max(s.warmup_periods() for s in self.strategies.values())


class _TaggedFactor(Factor):
    """Internal wrapper to tag factors with strategy name."""

    def __init__(self, wrapped_factor: Factor, tagged_name: str):
        super().__init__(tagged_name)
        self.wrapped_factor = wrapped_factor

    def compute(self, data):
        """Delegate to wrapped factor."""
        return self.wrapped_factor.compute(data)
