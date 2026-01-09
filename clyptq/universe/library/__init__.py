"""Prebuilt universe configurations.

Ready-to-use universes with common filter combinations.
All use operator-based filters only.

Example:
    ```python
    from clyptq.universe.library import CryptoLiquid, CryptoTop20

    # Use with top_n parameter
    universe = CryptoLiquid(top_n=30)

    # Or use convenience aliases
    universe = CryptoTop20()

    # Customize further
    universe = CryptoLiquid(
        top_n=50,
        min_dollar_volume=5e6,
        min_price=0.1
    )
    ```
"""

from clyptq.universe.library.crypto import (
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
    # Legacy aliases
    CryptoLiquid100,
    CryptoHighVolatility,
    CryptoLowVolatility,
)

__all__ = [
    # Main classes
    "CryptoLiquid",
    "CryptoVolatility",
    "CryptoStrata",
    # Liquidity aliases
    "CryptoTop10",
    "CryptoTop20",
    "CryptoTop30",
    "CryptoTop50",
    "CryptoTop100",
    # Volatility aliases
    "CryptoHighVol",
    "CryptoLowVol",
    "CryptoMidVol",
    # Strata helpers
    "CryptoL1Only",
    "CryptoL2Only",
    "CryptoDeFiOnly",
    "CryptoExcludeMeme",
    # Legacy aliases
    "CryptoLiquid100",
    "CryptoHighVolatility",
    "CryptoLowVolatility",
]
