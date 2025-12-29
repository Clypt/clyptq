"""
Multi-timeframe factor base class.

Enables factors to access and combine signals from multiple timeframes.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from clyptq.data.mtf_store import MultiTimeframeStore


class MultiTimeframeFactor(ABC):
    """
    Base class for factors using multiple timeframes.

    Subclasses must implement compute_mtf() which receives data
    from all requested timeframes.
    """

    def __init__(
        self,
        timeframes: List[str],
        lookbacks: Optional[Dict[str, int]] = None,
        name: Optional[str] = None,
    ):
        """
        Initialize multi-timeframe factor.

        Args:
            timeframes: List of timeframes to use (e.g., ["1d", "1w"])
            lookbacks: Dict mapping timeframe -> lookback periods
                      If None, uses default lookback for all
            name: Factor name (defaults to class name)
        """
        self.timeframes = timeframes
        self.lookbacks = lookbacks or {tf: 20 for tf in timeframes}
        self.name = name or self.__class__.__name__

        # Validate timeframes
        for tf in timeframes:
            if tf not in MultiTimeframeStore.VALID_TIMEFRAMES:
                raise ValueError(
                    f"Invalid timeframe '{tf}'. Must be one of {MultiTimeframeStore.VALID_TIMEFRAMES}"
                )

    def compute(
        self,
        mtf_store: MultiTimeframeStore,
        timestamp: datetime,
        symbols: List[str],
    ) -> Dict[str, float]:
        """
        Compute factor scores across multiple timeframes.

        Args:
            mtf_store: Multi-timeframe data store
            timestamp: Current timestamp
            symbols: List of symbols to compute

        Returns:
            Dict mapping symbol -> score
        """
        scores = {}

        for symbol in symbols:
            # Check if sufficient data exists for all timeframes
            has_data = all(
                mtf_store.has_sufficient_data(
                    symbol, tf, timestamp, self.lookbacks[tf]
                )
                for tf in self.timeframes
            )

            if not has_data:
                continue

            # Get data for each timeframe
            timeframe_data = {}
            for tf in self.timeframes:
                store = mtf_store.get_store(tf)
                data = store._data[symbol]
                available = data[data.index <= timestamp]

                # Get lookback window
                lookback = self.lookbacks[tf]
                window = available.iloc[-lookback:]
                timeframe_data[tf] = window

            # Compute score using multi-timeframe data
            score = self.compute_mtf(symbol, timeframe_data, timestamp)
            if score is not None:
                scores[symbol] = score

        return scores

    @abstractmethod
    def compute_mtf(
        self,
        symbol: str,
        timeframe_data: Dict[str, pd.DataFrame],
        timestamp: datetime,
    ) -> Optional[float]:
        """
        Compute factor score using data from multiple timeframes.

        Args:
            symbol: Trading symbol
            timeframe_data: Dict mapping timeframe -> OHLCV DataFrame
            timestamp: Current timestamp

        Returns:
            Factor score or None if cannot compute
        """
        pass

    def warmup_periods(self) -> int:
        """
        Get maximum warmup periods across all timeframes.

        Returns:
            Maximum lookback required
        """
        return max(self.lookbacks.values())

    def required_timeframes(self) -> List[str]:
        """
        Get list of required timeframes.

        Returns:
            List of timeframe strings
        """
        return self.timeframes.copy()


class MultiTimeframeMomentum(MultiTimeframeFactor):
    """
    Multi-timeframe momentum factor.

    Combines momentum signals from multiple timeframes:
    - Short-term: Fast momentum
    - Long-term: Trend confirmation

    Example:
        factor = MultiTimeframeMomentum(
            timeframes=["1d", "1w"],
            lookbacks={"1d": 20, "1w": 12}
        )
    """

    def __init__(
        self,
        timeframes: List[str] = ["1d", "1w"],
        lookbacks: Optional[Dict[str, int]] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize multi-timeframe momentum.

        Args:
            timeframes: List of timeframes
            lookbacks: Lookback periods for each timeframe
            weights: Weights for each timeframe (must sum to 1.0)
        """
        super().__init__(timeframes, lookbacks)

        # Default weights: equal weight
        if weights is None:
            weight = 1.0 / len(timeframes)
            self.weights = {tf: weight for tf in timeframes}
        else:
            self.weights = weights

        # Validate weights sum to 1
        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")

    def compute_mtf(
        self,
        symbol: str,
        timeframe_data: Dict[str, pd.DataFrame],
        timestamp: datetime,
    ) -> Optional[float]:
        """
        Compute weighted momentum across timeframes.

        Args:
            symbol: Trading symbol
            timeframe_data: Data for each timeframe
            timestamp: Current timestamp

        Returns:
            Combined momentum score
        """
        momentum_scores = {}

        for tf, data in timeframe_data.items():
            if len(data) < 2:
                return None

            # Calculate momentum: (latest - first) / first
            first_price = data.iloc[0]["close"]
            last_price = data.iloc[-1]["close"]

            if first_price <= 0:
                return None

            momentum = (last_price - first_price) / first_price
            momentum_scores[tf] = momentum

        # Weighted combination
        combined = sum(
            momentum_scores[tf] * self.weights[tf]
            for tf in self.timeframes
        )

        return combined
