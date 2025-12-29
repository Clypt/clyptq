"""
Portfolio construction strategies.

Transforms factor scores into target portfolio weights using different methods:
- TopN: Equal weight top N ranked assets
- ScoreWeighted: Weight proportional to scores
- RiskParity: Weight inversely proportional to volatility
"""

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np

from clyptq.core.types import Constraints


class PortfolioConstructor(ABC):
    """Abstract base class for portfolio construction."""

    @abstractmethod
    def construct(
        self, scores: Dict[str, float], constraints: Constraints
    ) -> Dict[str, float]:
        """
        Construct target portfolio from factor scores.

        Args:
            scores: Dictionary of {symbol: score}
            constraints: Portfolio constraints

        Returns:
            Dictionary of {symbol: weight} where sum(weights) <= max_gross_exposure
        """
        pass


class TopNConstructor(PortfolioConstructor):
    """
    Top-N equal-weight portfolio constructor.

    Selects top N highest-scoring assets and assigns equal weights.
    """

    def __init__(self, top_n: int = 10):
        """
        Initialize TopN constructor.

        Args:
            top_n: Number of top assets to select
        """
        if top_n <= 0:
            raise ValueError(f"top_n must be positive, got {top_n}")
        self.top_n = top_n

    def construct(
        self, scores: Dict[str, float], constraints: Constraints
    ) -> Dict[str, float]:
        """
        Construct equal-weight portfolio of top N assets.

        Args:
            scores: Dictionary of {symbol: score}
            constraints: Portfolio constraints

        Returns:
            Dictionary of {symbol: weight}
        """
        if not scores:
            return {}

        # Sort by score descending, then by symbol name for determinism
        sorted_symbols = sorted(scores.items(), key=lambda x: (-x[1], x[0]))

        # Select top N (or fewer if not enough symbols)
        n = min(self.top_n, len(sorted_symbols), constraints.max_num_positions)

        # Equal weight
        weight = min(
            constraints.max_position_size, constraints.max_gross_exposure / n
        )

        # Assign weights
        weights = {}
        for symbol, score in sorted_symbols[:n]:
            if weight >= constraints.min_position_size:
                weights[symbol] = weight

        # Normalize to respect max_gross_exposure
        total_weight = sum(weights.values())
        if total_weight > constraints.max_gross_exposure:
            scale = constraints.max_gross_exposure / total_weight
            weights = {s: w * scale for s, w in weights.items()}

        return weights


class ScoreWeightedConstructor(PortfolioConstructor):
    """
    Score-weighted portfolio constructor.

    Weights are proportional to (normalized) factor scores.
    """

    def __init__(self, use_long_short: bool = False):
        """
        Initialize score-weighted constructor.

        Args:
            use_long_short: If True, negative scores become short positions
        """
        self.use_long_short = use_long_short

    def construct(
        self, scores: Dict[str, float], constraints: Constraints
    ) -> Dict[str, float]:
        """
        Construct score-weighted portfolio.

        Args:
            scores: Dictionary of {symbol: score}
            constraints: Portfolio constraints

        Returns:
            Dictionary of {symbol: weight}
        """
        if not scores:
            return {}

        # Filter and normalize scores
        if self.use_long_short and constraints.allow_short:
            # Keep all scores, normalize by sum of absolute values
            symbols = sorted(scores.keys())
            score_values = np.array([scores[s] for s in symbols])

            # Handle case where all scores are zero
            abs_sum = np.abs(score_values).sum()
            if abs_sum < 1e-10:
                return {}

            # Normalize
            raw_weights = score_values / abs_sum * constraints.max_gross_exposure

        else:
            # Long-only: filter positive scores
            positive_scores = {s: sc for s, sc in scores.items() if sc > 0}
            if not positive_scores:
                return {}

            symbols = sorted(positive_scores.keys())
            score_values = np.array([positive_scores[s] for s in symbols])

            # Normalize
            score_sum = score_values.sum()
            if score_sum < 1e-10:
                return {}

            raw_weights = score_values / score_sum * constraints.max_gross_exposure

        # Apply position size constraints
        weights = {}
        for symbol, weight in zip(symbols, raw_weights):
            # Clip to max position size
            if weight > 0:
                weight = min(weight, constraints.max_position_size)
            elif weight < 0:
                weight = max(weight, -constraints.max_position_size)

            # Only include if above minimum
            if abs(weight) >= constraints.min_position_size:
                weights[symbol] = weight

        # Limit number of positions
        if len(weights) > constraints.max_num_positions:
            # Keep largest absolute weights, then by symbol name for determinism
            sorted_weights = sorted(
                weights.items(), key=lambda x: (-abs(x[1]), x[0])
            )
            weights = dict(sorted_weights[: constraints.max_num_positions])

        # Normalize to respect max_gross_exposure
        total_abs_weight = sum(abs(w) for w in weights.values())
        if total_abs_weight > constraints.max_gross_exposure:
            scale = constraints.max_gross_exposure / total_abs_weight
            weights = {s: w * scale for s, w in weights.items()}

        return weights


class RiskParityConstructor(PortfolioConstructor):
    """
    Risk parity portfolio constructor.

    Weights inversely proportional to volatility (equal risk contribution).
    Requires volatility estimates for each asset.
    """

    def __init__(self, volatility_lookback: int = 20, min_volatility: float = 0.001):
        """
        Initialize risk parity constructor.

        Args:
            volatility_lookback: Lookback period for volatility estimation
            min_volatility: Minimum volatility to prevent division by zero
        """
        self.volatility_lookback = volatility_lookback
        self.min_volatility = min_volatility
        self._volatility_cache: Dict[str, float] = {}

    def set_volatility(self, symbol: str, volatility: float) -> None:
        """
        Set volatility estimate for a symbol.

        Args:
            symbol: Symbol identifier
            volatility: Volatility estimate (annualized)
        """
        self._volatility_cache[symbol] = max(volatility, self.min_volatility)

    def set_volatilities(self, volatilities: Dict[str, float]) -> None:
        """
        Set volatility estimates for multiple symbols.

        Args:
            volatilities: Dictionary of {symbol: volatility}
        """
        for symbol, vol in volatilities.items():
            self.set_volatility(symbol, vol)

    def construct(
        self, scores: Dict[str, float], constraints: Constraints
    ) -> Dict[str, float]:
        """
        Construct risk parity portfolio.

        Positive scores are selected, then weighted inversely to volatility.

        Args:
            scores: Dictionary of {symbol: score}
            constraints: Portfolio constraints

        Returns:
            Dictionary of {symbol: weight}

        Raises:
            ValueError: If volatilities not set for all symbols
        """
        if not scores:
            return {}

        # Filter positive scores (long-only for risk parity)
        positive_scores = {s: sc for s, sc in scores.items() if sc > 0}
        if not positive_scores:
            return {}

        # Check volatility cache
        symbols = sorted(positive_scores.keys())
        missing = [s for s in symbols if s not in self._volatility_cache]
        if missing:
            raise ValueError(
                f"Volatilities not set for symbols: {missing}. "
                f"Use set_volatility() or set_volatilities() first."
            )

        # Inverse volatility weights
        inv_vols = np.array([1.0 / self._volatility_cache[s] for s in symbols])
        raw_weights = inv_vols / inv_vols.sum() * constraints.max_gross_exposure

        # Apply position size constraints
        weights = {}
        for symbol, weight in zip(symbols, raw_weights):
            # Clip to max position size
            weight = min(weight, constraints.max_position_size)

            # Only include if above minimum
            if weight >= constraints.min_position_size:
                weights[symbol] = weight

        # Limit number of positions (keep largest weights), then by symbol name for determinism
        if len(weights) > constraints.max_num_positions:
            sorted_weights = sorted(weights.items(), key=lambda x: (-x[1], x[0]))
            weights = dict(sorted_weights[: constraints.max_num_positions])

        # Normalize to respect max_gross_exposure
        total_weight = sum(weights.values())
        if total_weight > constraints.max_gross_exposure:
            scale = constraints.max_gross_exposure / total_weight
            weights = {s: w * scale for s, w in weights.items()}

        return weights


class BlendedConstructor(PortfolioConstructor):
    """
    Blended portfolio constructor for multi-strategy allocation.

    Runs multiple strategies independently and blends their target weights
    according to allocation percentages.
    """

    def __init__(
        self,
        strategies: Dict[str, "Strategy"],
        allocations: Dict[str, float],
        factor_map: Dict[str, str],
    ):
        """
        Initialize blended constructor.

        Args:
            strategies: Dict of {strategy_name: Strategy}
            allocations: Dict of {strategy_name: allocation_weight}
            factor_map: Dict of {factor_name: strategy_name}
        """
        if set(strategies.keys()) != set(allocations.keys()):
            raise ValueError("Strategies and allocations must have same keys")

        total_alloc = sum(allocations.values())
        if abs(total_alloc - 1.0) > 1e-6:
            raise ValueError(f"Allocations must sum to 1.0, got {total_alloc}")

        self.strategies = strategies
        self.allocations = allocations
        self.factor_map = factor_map

    def construct(
        self, scores: Dict[str, float], constraints: Constraints
    ) -> Dict[str, float]:
        """
        Construct blended portfolio from multiple strategies.

        Each strategy uses the combined scores but applies its own constructor
        and constraints. The resulting weights are blended according to allocations.

        Args:
            scores: Combined scores from all factors (symbol: score)
            constraints: Portfolio constraints

        Returns:
            Blended target weights
        """
        if not scores:
            return {}

        strategy_weights: Dict[str, Dict[str, float]] = {}

        for strategy_name, strategy in self.strategies.items():
            constructor = strategy.portfolio_constructor()
            strategy_constraints = strategy.constraints()
            weights = constructor.construct(scores, strategy_constraints)
            strategy_weights[strategy_name] = weights

        blended_weights: Dict[str, float] = {}
        for strategy_name, weights in strategy_weights.items():
            allocation = self.allocations[strategy_name]
            for symbol, weight in weights.items():
                blended_weights[symbol] = (
                    blended_weights.get(symbol, 0.0) + weight * allocation
                )

        total_weight = sum(abs(w) for w in blended_weights.values())
        if total_weight > constraints.max_gross_exposure:
            scale = constraints.max_gross_exposure / total_weight
            blended_weights = {s: w * scale for s, w in blended_weights.items()}

        return blended_weights
