"""
Base classes for alpha factor computation.

Provides abstract base class for factors and factor combination utilities.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from clypt.data.store import DataView


class Factor(ABC):
    """
    Abstract base class for alpha factors.

    All factors must implement the compute() method which takes a DataView
    and returns a dictionary of {symbol: score}.
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialize factor.

        Args:
            name: Factor name (defaults to class name)
        """
        self.name = name or self.__class__.__name__

    @abstractmethod
    def compute(self, data: DataView) -> Dict[str, float]:
        """
        Compute factor scores for all available symbols.

        IMPORTANT: Omit symbols with insufficient data rather than
        returning NaN or zero scores.

        Args:
            data: DataView at current timestamp

        Returns:
            Dictionary of {symbol: score}
            Symbols with insufficient data should be omitted
        """
        pass

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name='{self.name}')"


class CombinedFactor(Factor):
    """
    Combines multiple factors using weighted average.

    Useful for creating composite signals from multiple alpha factors.
    """

    def __init__(
        self, factors: List[Factor], weights: Optional[List[float]] = None, name: Optional[str] = None
    ):
        """
        Initialize combined factor.

        Args:
            factors: List of factors to combine
            weights: Optional weights for each factor (defaults to equal weight)
            name: Combined factor name
        """
        super().__init__(name or "Combined")

        if not factors:
            raise ValueError("Must provide at least one factor")

        self.factors = factors

        # Set equal weights if not provided
        if weights is None:
            weights = [1.0 / len(factors)] * len(factors)

        if len(weights) != len(factors):
            raise ValueError(
                f"Number of weights ({len(weights)}) must match "
                f"number of factors ({len(factors)})"
            )

        # Normalize weights to sum to 1
        total_weight = sum(weights)
        self.weights = [w / total_weight for w in weights]

    def compute(self, data: DataView) -> Dict[str, float]:
        """
        Compute combined factor scores.

        For each symbol, computes weighted average of individual factor scores.
        Only includes symbols that have scores from at least one factor.

        Args:
            data: DataView at current timestamp

        Returns:
            Dictionary of {symbol: combined_score}
        """
        # Compute all factor scores
        all_scores: List[Dict[str, float]] = []
        for factor in self.factors:
            scores = factor.compute(data)
            all_scores.append(scores)

        # Get all symbols across all factors
        all_symbols = set()
        for scores in all_scores:
            all_symbols.update(scores.keys())

        # Combine scores
        combined = {}
        for symbol in all_symbols:
            total_score = 0.0
            total_weight = 0.0

            for scores, weight in zip(all_scores, self.weights):
                if symbol in scores:
                    total_score += scores[symbol] * weight
                    total_weight += weight

            # Only include if at least one factor had a score
            if total_weight > 0:
                combined[symbol] = total_score / total_weight

        return combined


class RankedFactor(Factor):
    """
    Wrapper that converts raw factor values to cross-sectional ranks.

    Useful for normalizing factor exposures and combining factors.
    """

    def __init__(self, base_factor: Factor, normalize: bool = True):
        """
        Initialize ranked factor.

        Args:
            base_factor: Underlying factor to rank
            normalize: If True, normalize ranks to [-1, 1]
        """
        super().__init__(name=f"Ranked_{base_factor.name}")
        self.base_factor = base_factor
        self.normalize = normalize

    def compute(self, data: DataView) -> Dict[str, float]:
        """
        Compute cross-sectional ranks.

        Args:
            data: DataView at current timestamp

        Returns:
            Dictionary of {symbol: rank}
            Ranks are integers starting from 0, or normalized to [-1, 1]
        """
        # Get raw scores
        raw_scores = self.base_factor.compute(data)

        if not raw_scores:
            return {}

        # Sort by score
        sorted_symbols = sorted(raw_scores.items(), key=lambda x: x[1])

        # Assign ranks
        ranked = {}
        for rank, (symbol, _) in enumerate(sorted_symbols):
            if self.normalize:
                # Normalize to [-1, 1]
                n = len(sorted_symbols)
                normalized_rank = 2.0 * rank / (n - 1) - 1.0 if n > 1 else 0.0
                ranked[symbol] = normalized_rank
            else:
                ranked[symbol] = float(rank)

        return ranked


class ZScoreFactor(Factor):
    """
    Wrapper that z-score normalizes factor values.

    Subtracts mean and divides by standard deviation for cross-sectional normalization.
    """

    def __init__(self, base_factor: Factor, min_symbols: int = 5):
        """
        Initialize z-score factor.

        Args:
            base_factor: Underlying factor to normalize
            min_symbols: Minimum symbols required for normalization
        """
        super().__init__(name=f"ZScore_{base_factor.name}")
        self.base_factor = base_factor
        self.min_symbols = min_symbols

    def compute(self, data: DataView) -> Dict[str, float]:
        """
        Compute z-score normalized values.

        Args:
            data: DataView at current timestamp

        Returns:
            Dictionary of {symbol: z_score}
        """
        # Get raw scores
        raw_scores = self.base_factor.compute(data)

        if len(raw_scores) < self.min_symbols:
            return {}

        # Calculate mean and std
        values = list(raw_scores.values())
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = variance**0.5

        # Handle zero std
        if std < 1e-10:
            return {}

        # Z-score normalize
        z_scores = {symbol: (score - mean) / std for symbol, score in raw_scores.items()}

        return z_scores


def combine_factors(
    factors: List[Factor], weights: Optional[List[float]] = None, name: Optional[str] = None
) -> CombinedFactor:
    """
    Convenience function to create a combined factor.

    Args:
        factors: List of factors to combine
        weights: Optional weights for each factor
        name: Combined factor name

    Returns:
        CombinedFactor instance
    """
    return CombinedFactor(factors, weights, name)
