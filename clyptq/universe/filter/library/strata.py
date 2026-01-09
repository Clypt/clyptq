"""Strata-based filter using operator-based computation."""

from typing import Dict, List, Optional, Set

from clyptq import operator
from clyptq.universe.filter.base import BaseFilter


class StrataFilter(BaseFilter):
    """Filter symbols by strata (group/category) membership.

    Keep only symbols belonging to specified strata categories.

    Note: This filter uses strata dict (symbol -> category) which is
    static metadata provided at init time. The actual computation
    uses operators only.

    Example:
        ```python
        # Only trade L1 and L2 tokens
        filter = StrataFilter(
            strata={"BTC": "L1", "ETH": "L1", "ARB": "L2", ...},
            include=["L1", "L2"]
        )

        # Exclude meme coins
        filter = StrataFilter(
            strata=provider.get_strata("layer"),
            exclude=["Meme"]
        )
        ```
    """

    def __init__(
        self,
        strata: Dict[str, str],
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        name: Optional[str] = None,
    ):
        """Initialize strata filter.

        Args:
            strata: Dict mapping symbol -> category
            include: Categories to include (None = include all)
            exclude: Categories to exclude (applied after include)
            name: Optional identifier
        """
        super().__init__(name or "StrataFilter")
        self._strata = strata
        self._include: Optional[Set[str]] = set(include) if include else None
        self._exclude: Set[str] = set(exclude) if exclude else set()

        # Precompute which symbols pass the filter (static)
        self._passing_symbols: Set[str] = set()
        for symbol, category in strata.items():
            if self._include is not None and category not in self._include:
                continue
            if category in self._exclude:
                continue
            self._passing_symbols.add(symbol)

    def compute(self, data):
        """Compute strata filter mask.

        Args:
            data: DataProvider with market data

        Returns:
            Boolean mask (T x N) - True = passes filter
        """
        close = data["close"]

        # Start with valid data mask
        valid_mask = operator.notna(close)

        # Create strata mask using operator.broadcast_mask
        # For symbols not in passing_symbols, set to False
        strata_mask = operator.broadcast_mask(close, self._passing_symbols)

        # Combine: must have valid data AND pass strata filter
        mask = operator.logical_and(valid_mask, strata_mask)

        return mask

    def __repr__(self) -> str:
        inc = f", include={list(self._include)}" if self._include else ""
        exc = f", exclude={list(self._exclude)}" if self._exclude else ""
        return f"StrataFilter({inc}{exc})"
