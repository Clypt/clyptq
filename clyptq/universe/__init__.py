"""Universe management for strategy execution.

This module provides:
- Universe: Unified universe with filters, scoring, and topN
- StaticUniverse: Fixed list of symbols
- DynamicUniverse: Time-varying universe based on filters
- Filters: LiquidityFilter, VolumeFilter, PriceFilter, etc.
- Prebuilt universes: CryptoLiquid, CryptoTop20, CryptoTop50, etc.
- load_universe: Load data from universe as wide DataFrame for analysis

Architecture:
    All symbols (N) → Filters → Tradeable (M) → Scoring+TopN → In Universe (n)

Usage:
    ```python
    from clyptq.universe import CryptoLiquid, LiquidityFilter

    # Use prebuilt universe with top_n
    universe = CryptoLiquid(top_n=30)

    # Or use convenience alias
    from clyptq.universe import CryptoTop20
    universe = CryptoTop20()

    # Or build custom universe
    universe = Universe(
        filters=[LiquidityFilter(min_dollar_volume=1e6)],
        n=50
    )
    ```

Structure:
    universe/
    ├── base.py           # Universe, StaticUniverse, DynamicUniverse
    ├── loader.py         # load_universe, load_universe_dynamic
    ├── filter/
    │   ├── base.py       # BaseFilter abstract class
    │   └── library/      # Operator-based filter implementations
    └── library/          # Prebuilt universe configurations
        └── crypto.py
"""

# Core universe classes
from clyptq.universe.base import (
    Universe,
    StaticUniverse,
    DynamicUniverse,
)

# Filter base and implementations
from clyptq.universe.filter import (
    BaseFilter,
    LiquidityFilter,
    VolumeFilter,
    PriceFilter,
    VolatilityFilter,
    DataAvailabilityFilter,
    CompositeFilter,
    StrataFilter,
)

# Prebuilt universes
from clyptq.universe.library import (
    # Main classes
    CryptoLiquid,
    CryptoVolatility,
    CryptoStrata,
    # Liquidity aliases
    CryptoTop10,
    CryptoTop20,
    CryptoTop30,
    CryptoTop50,
    CryptoTop100,
    # Volatility aliases
    CryptoHighVol,
    CryptoLowVol,
    CryptoMidVol,
    # Strata helpers
    CryptoL1Only,
    CryptoL2Only,
    CryptoDeFiOnly,
    CryptoExcludeMeme,
    # Legacy
    CryptoLiquid100,
    CryptoHighVolatility,
    CryptoLowVolatility,
)

# Loader utilities
from clyptq.universe.loader import (
    load_universe,
    load_universe_dynamic,
    filter_to_target,
    UniverseData,
)

# Legacy aliases for backward compatibility
UniverseFilter = BaseFilter  # Deprecated: use BaseFilter
TopNFilter = None  # Deprecated: use Universe.n parameter
StrataBalancedFilter = None  # Deprecated: use Universe with scoring
StrataProportionalFilter = None  # Deprecated: use Universe with scoring

__all__ = [
    # Core
    "Universe",
    "StaticUniverse",
    "DynamicUniverse",
    # Filter base
    "BaseFilter",
    # Filters
    "LiquidityFilter",
    "VolumeFilter",
    "PriceFilter",
    "VolatilityFilter",
    "DataAvailabilityFilter",
    "CompositeFilter",
    "StrataFilter",
    # Prebuilt universes - main classes
    "CryptoLiquid",
    "CryptoVolatility",
    "CryptoStrata",
    # Prebuilt universes - liquidity aliases
    "CryptoTop10",
    "CryptoTop20",
    "CryptoTop30",
    "CryptoTop50",
    "CryptoTop100",
    # Prebuilt universes - volatility aliases
    "CryptoHighVol",
    "CryptoLowVol",
    "CryptoMidVol",
    # Prebuilt universes - strata helpers
    "CryptoL1Only",
    "CryptoL2Only",
    "CryptoDeFiOnly",
    "CryptoExcludeMeme",
    # Legacy aliases
    "CryptoLiquid100",
    "CryptoHighVolatility",
    "CryptoLowVolatility",
    # Loader
    "load_universe",
    "load_universe_dynamic",
    "filter_to_target",
    "UniverseData",
    # Legacy (deprecated)
    "UniverseFilter",
    "TopNFilter",
    "StrataBalancedFilter",
    "StrataProportionalFilter",
]
