"""Base universe classes for strategy execution.

Universe defines the set of tradeable symbols at any given time.
This is critical for:
- Pasteurize: Determining N (global) vs n (target) universe
- Position limits: How many symbols can be held
- Alpha computation: Which symbols to compute scores for

Architecture:
    All symbols (N) → Filters → Tradeable (M) → Scoring+TopN → In Universe (n)

    - N: Total available symbols from data source
    - M: Symbols passing filters (data loaded for these)
    - n: Symbols in target universe (pasteurize applies to these)
"""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Union

import pandas as pd

from clyptq import operator
from clyptq.infra.utils import get_logger

if TYPE_CHECKING:
    from clyptq.data.provider import DataProvider
    from clyptq.strategy.signal.base import BaseSignal
    from clyptq.universe.filter.base import BaseFilter

logger = get_logger(__name__)


class Universe:
    """Unified universe with filters, scoring, and topN.

    Example:
        ```python
        from clyptq.universe import Universe
        from clyptq.universe.filter import LiquidityFilter, PriceFilter

        universe = Universe(
            filters=[
                LiquidityFilter(min_dollar_volume=1e6),
                PriceFilter(min_price=1.0),
            ],
            scoring=VolumeAlpha(lookback=20),  # or callable
            n=50,
            rebalance_freq="1d",
        )
        ```

    Flow at each rebalance:
        1. compute_tradeable(): Apply filters → M symbols (data load target)
        2. compute_in_universe(): Apply scoring+topN → n symbols (pasteurize target)
    """

    def __init__(
        self,
        filters: Optional[List["BaseFilter"]] = None,
        scoring: Optional[Union["BaseSignal", Callable]] = None,
        n: Optional[int] = None,
        rebalance_freq: str = "1d",
        name: Optional[str] = None,
    ):
        """Initialize Universe.

        Args:
            filters: List of filters to apply (determines tradeable set M)
            scoring: Signal or callable for ranking symbols
            n: Number of symbols in target universe (None = all tradeable)
            rebalance_freq: How often to update universe membership
            name: Universe name
        """
        self.filters = filters or []
        self.scoring = scoring
        self.n = n
        self.rebalance_freq = rebalance_freq
        self.name = name or "Universe"

        # Cache for efficiency
        self._tradeable_mask: Optional[pd.DataFrame] = None
        self._in_universe_mask: Optional[pd.DataFrame] = None
        self._last_update_ts: Optional[datetime] = None

    def compute_tradeable(self, data: "DataProvider") -> pd.DataFrame:
        """Compute tradeable symbols mask (M symbols).

        Applies all filters. Returns boolean DataFrame.

        Args:
            data: DataProvider with market data

        Returns:
            Boolean DataFrame (T x N) - True = tradeable
        """
        close = data["close"]

        # Start with all symbols as tradeable (non-NaN)
        mask = operator.notna(close)

        # Apply each filter (AND logic)
        for filter_ in self.filters:
            filter_mask = filter_.compute(data)
            mask = operator.logical_and(mask, filter_mask)

            logger.debug(
                f"[Universe] Filter {filter_.name}: "
                f"{mask.iloc[-1].sum()}/{len(mask.columns)} symbols pass"
            )

        self._tradeable_mask = mask
        return mask

    def compute_in_universe(self, data: "DataProvider") -> pd.DataFrame:
        """Compute in_universe mask (n symbols).

        Applies scoring and topN selection.

        Args:
            data: DataProvider with market data

        Returns:
            Boolean DataFrame (T x N) - True = in universe
        """
        # Get tradeable mask
        if self._tradeable_mask is None:
            self._tradeable_mask = self.compute_tradeable(data)

        tradeable = self._tradeable_mask

        # If no scoring or n, all tradeable symbols are in universe
        if self.scoring is None or self.n is None:
            self._in_universe_mask = tradeable
            return tradeable

        # Compute scores
        if callable(self.scoring) and not hasattr(self.scoring, 'compute'):
            # Lambda or function
            scores = self.scoring(data)
        else:
            # BaseSignal
            scores = self.scoring.compute(data)

        # Mask non-tradeable symbols
        scores = operator.where(tradeable, scores, float('nan'))

        # TopN selection per row
        in_universe = self._topn_mask(scores, self.n)

        self._in_universe_mask = in_universe
        self._last_update_ts = data.current_timestamp

        logger.debug(
            f"[Universe] TopN({self.n}): "
            f"{in_universe.iloc[-1].sum()}/{len(in_universe.columns)} symbols in universe"
        )

        return in_universe

    def _topn_mask(self, scores: pd.DataFrame, n: int) -> pd.DataFrame:
        """Create boolean mask for top N symbols per row.

        Args:
            scores: Score DataFrame (T x N)
            n: Number of top symbols to select

        Returns:
            Boolean DataFrame (T x N)
        """
        # Rank scores (higher = better)
        ranked = operator.rank(scores)

        # Count valid symbols per row
        valid_count = operator.notna(scores).sum(axis=1)

        # Threshold for top N (percentile cutoff)
        # rank returns 0-1 percentile, so top N means rank >= (1 - n/valid_count)
        threshold = operator.sub(1, operator.div(n, valid_count))

        # Create mask: rank >= threshold
        mask = pd.DataFrame(index=scores.index, columns=scores.columns, dtype=bool)
        for i, ts in enumerate(scores.index):
            row_scores = ranked.iloc[i]
            row_threshold = threshold.iloc[i] if hasattr(threshold, 'iloc') else threshold
            mask.iloc[i] = row_scores >= row_threshold

        return mask

    def get_tradeable_symbols(self, data: "DataProvider") -> List[str]:
        """Get list of tradeable symbols at current timestamp.

        Args:
            data: DataProvider

        Returns:
            List of tradeable symbol names
        """
        if self._tradeable_mask is None:
            self.compute_tradeable(data)

        mask = self._tradeable_mask
        if mask is None or mask.empty:
            return []

        current_mask = mask.iloc[-1]
        return current_mask[current_mask].index.tolist()

    def get_in_universe_symbols(self, data: "DataProvider") -> List[str]:
        """Get list of in_universe symbols at current timestamp.

        Args:
            data: DataProvider

        Returns:
            List of in_universe symbol names
        """
        if self._in_universe_mask is None:
            self.compute_in_universe(data)

        mask = self._in_universe_mask
        if mask is None or mask.empty:
            return []

        current_mask = mask.iloc[-1]
        return current_mask[current_mask].index.tolist()

    def get_in_universe(self, timestamp: datetime) -> List[str]:
        """Get in_universe symbols at specific timestamp.

        Args:
            timestamp: Target timestamp

        Returns:
            List of symbol names in universe at that timestamp

        Raises:
            ValueError: If compute_in_universe() not called yet
        """
        if self._in_universe_mask is None:
            raise ValueError("compute_in_universe() must be called first")

        if timestamp not in self._in_universe_mask.index:
            # Find nearest timestamp
            idx = self._in_universe_mask.index.get_indexer([timestamp], method="ffill")[0]
            if idx < 0:
                return []
            timestamp = self._in_universe_mask.index[idx]

        mask_at_t = self._in_universe_mask.loc[timestamp]
        return mask_at_t[mask_at_t].index.tolist()

    def get_in_universe_mask(self, timestamp: datetime) -> pd.Series:
        """Get in_universe boolean mask at specific timestamp.

        Args:
            timestamp: Target timestamp

        Returns:
            Boolean Series (N symbols) - True = in universe

        Raises:
            ValueError: If compute_in_universe() not called yet
        """
        if self._in_universe_mask is None:
            raise ValueError("compute_in_universe() must be called first")

        if timestamp not in self._in_universe_mask.index:
            # Find nearest timestamp
            idx = self._in_universe_mask.index.get_indexer([timestamp], method="ffill")[0]
            if idx < 0:
                return pd.Series(dtype=bool)
            timestamp = self._in_universe_mask.index[idx]

        return self._in_universe_mask.loc[timestamp]

    def reset(self) -> None:
        """Reset cached state."""
        self._tradeable_mask = None
        self._in_universe_mask = None
        self._last_update_ts = None

    def __repr__(self) -> str:
        return (
            f"Universe(filters={len(self.filters)}, "
            f"n={self.n}, rebalance={self.rebalance_freq})"
        )


class StaticUniverse(Universe):
    """Fixed universe with predefined symbols.

    Use when universe doesn't change over time.

    Example:
        ```python
        universe = StaticUniverse(["BTC", "ETH", "SOL", "AVAX"])
        ```
    """

    def __init__(self, symbols: List[str], name: Optional[str] = None):
        """Initialize static universe.

        Args:
            symbols: Fixed list of symbols
            name: Optional identifier
        """
        super().__init__(name=name or "StaticUniverse")
        self._symbols = list(symbols)
        self._symbol_set = set(symbols)

    @property
    def symbols(self) -> List[str]:
        """Get symbol list."""
        return self._symbols

    def compute_tradeable(self, data: "DataProvider") -> pd.DataFrame:
        """All static symbols are tradeable."""
        close = data["close"]

        # Filter to static symbols that exist in data
        available_cols = [s for s in self._symbols if s in close.columns]

        # Create mask: True for static symbols, False otherwise
        mask = pd.DataFrame(
            False,
            index=close.index,
            columns=close.columns,
        )
        for col in available_cols:
            mask[col] = operator.notna(close[col])

        self._tradeable_mask = mask
        return mask

    def compute_in_universe(self, data: "DataProvider") -> pd.DataFrame:
        """All static symbols are in universe."""
        return self.compute_tradeable(data)


class DynamicUniverse(Universe):
    """Time-varying universe based on filters.

    Legacy wrapper around Universe for backward compatibility.
    """

    def __init__(
        self,
        base_symbols: Optional[List[str]] = None,
        filters: Optional[List["BaseFilter"]] = None,
        max_symbols: Optional[int] = None,
        name: Optional[str] = None,
    ):
        """Initialize dynamic universe.

        Args:
            base_symbols: Starting pool of symbols (None = all available)
            filters: List of filters to apply
            max_symbols: Maximum number of symbols to include
            name: Optional identifier
        """
        super().__init__(
            filters=filters or [],
            n=max_symbols,
            name=name or "DynamicUniverse",
        )
        self._base_symbols = base_symbols

    @property
    def symbols(self) -> Optional[List[str]]:
        """Get base symbol list."""
        return self._base_symbols

    # Legacy method for backward compatibility
    def get_symbols(
        self,
        timestamp: datetime,
        source: Any,
    ) -> List[str]:
        """Legacy method - returns base symbols."""
        if self._base_symbols:
            return self._base_symbols
        if hasattr(source, 'available_symbols'):
            return source.available_symbols()
        return []


# Legacy alias
BaseUniverse = Universe
