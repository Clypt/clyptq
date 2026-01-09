"""Universe filters for symbol selection.

Filters return boolean masks indicating which symbols pass criteria.
All filters MUST use clyptq operators only - no pandas/numpy direct usage.

Example:
    ```python
    from clyptq.universe.filter import LiquidityFilter, PriceFilter, CompositeFilter

    # Single filter
    liquidity = LiquidityFilter(min_dollar_volume=1e6, lookback=20)

    # Combine filters
    combined = CompositeFilter([
        LiquidityFilter(min_dollar_volume=1e6),
        PriceFilter(min_price=5.0),
    ])
    ```
"""

from clyptq.universe.filter.base import BaseFilter

# Import all filters from library
from clyptq.universe.filter.library import (
    LiquidityFilter,
    VolumeFilter,
    PriceFilter,
    VolatilityFilter,
    DataAvailabilityFilter,
    CompositeFilter,
    StrataFilter,
)

__all__ = [
    "BaseFilter",
    "LiquidityFilter",
    "VolumeFilter",
    "PriceFilter",
    "VolatilityFilter",
    "DataAvailabilityFilter",
    "CompositeFilter",
    "StrataFilter",
]
