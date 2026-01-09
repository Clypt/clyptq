"""Prebuilt crypto universe configurations.

All universes use operator-based filters only.

Usage:
    ```python
    from clyptq.universe import CryptoLiquid

    # Default: top 50 by liquidity
    universe = CryptoLiquid()

    # Customize top N
    universe = CryptoLiquid(top_n=20)

    # Customize filters
    universe = CryptoLiquid(
        top_n=30,
        min_dollar_volume=5e6,
        min_price=0.1
    )
    ```
"""

from typing import Optional

from clyptq.universe.base import Universe
from clyptq.universe.filter import (
    LiquidityFilter,
    PriceFilter,
    VolatilityFilter,
    DataAvailabilityFilter,
    StrataFilter,
)


class CryptoLiquid(Universe):
    """Liquid crypto universe by dollar volume.

    Default: Top 50 by average daily dollar volume.

    Args:
        top_n: Number of top symbols to include (default: 50)
        min_dollar_volume: Minimum dollar volume threshold (default: $1M)
        min_price: Minimum price threshold (default: $0.01)
        min_bars: Minimum data bars required (default: 20)
        lookback: Lookback period for liquidity calculation (default: 20)
        rebalance_freq: Rebalance frequency (default: "1d")

    Example:
        ```python
        # Top 50 liquid crypto
        universe = CryptoLiquid()

        # Top 20 with higher liquidity threshold
        universe = CryptoLiquid(top_n=20, min_dollar_volume=5e6)

        # Top 30 with custom lookback
        universe = CryptoLiquid(top_n=30, min_dollar_volume=100_000, lookback=10)
        ```
    """

    def __init__(
        self,
        top_n: int = 50,
        min_dollar_volume: float = 1_000_000,
        min_price: float = 0.01,
        min_bars: int = 20,
        lookback: int = 20,
        rebalance_freq: str = "1d",
    ):
        name = f"CryptoLiquid{top_n}"
        self._lookback = lookback

        # Scoring function: average dollar volume (higher = better)
        def dollar_volume_score(data):
            from clyptq import operator
            close = data["close"]
            volume = data["volume"]
            dollar_vol = operator.mul(close, volume)
            return operator.ts_mean(dollar_vol, lookback)

        super().__init__(
            filters=[
                LiquidityFilter(min_dollar_volume=min_dollar_volume, lookback=lookback),
                PriceFilter(min_price=min_price),
                DataAvailabilityFilter(min_bars=min_bars),
            ],
            scoring=dollar_volume_score,
            n=top_n,
            rebalance_freq=rebalance_freq,
            name=name,
        )


class CryptoVolatility(Universe):
    """Crypto universe filtered by volatility range.

    Args:
        top_n: Number of top symbols (None = all passing filters)
        min_vol: Minimum annualized volatility (default: 0.1 = 10%)
        max_vol: Maximum annualized volatility (default: 3.0 = 300%)
        min_dollar_volume: Minimum dollar volume (default: $1M)
        rebalance_freq: Rebalance frequency

    Example:
        ```python
        # High volatility: 50% - 300% annualized
        universe = CryptoVolatility(min_vol=0.5, max_vol=3.0)

        # Low volatility: 10% - 80% annualized
        universe = CryptoVolatility(min_vol=0.1, max_vol=0.8)

        # Top 30 medium volatility
        universe = CryptoVolatility(top_n=30, min_vol=0.3, max_vol=1.0)
        ```
    """

    def __init__(
        self,
        top_n: Optional[int] = None,
        min_vol: float = 0.1,
        max_vol: float = 3.0,
        min_dollar_volume: float = 1_000_000,
        min_bars: int = 30,
        lookback: int = 20,
        rebalance_freq: str = "1d",
    ):
        vol_desc = f"Vol{int(min_vol*100)}-{int(max_vol*100)}"
        n_desc = f"Top{top_n}" if top_n else "All"
        name = f"Crypto{vol_desc}_{n_desc}"

        super().__init__(
            filters=[
                LiquidityFilter(min_dollar_volume=min_dollar_volume, lookback=lookback),
                VolatilityFilter(min_vol=min_vol, max_vol=max_vol, lookback=lookback),
                DataAvailabilityFilter(min_bars=min_bars),
            ],
            n=top_n,
            rebalance_freq=rebalance_freq,
            name=name,
        )


class CryptoStrata(Universe):
    """Crypto universe filtered by strata (sector/category).

    Requires strata dict mapping symbol -> category.

    Args:
        strata: Dict mapping symbol to category
        include: Categories to include (e.g., ["L1", "L2"])
        exclude: Categories to exclude (e.g., ["Meme"])
        top_n: Number of top symbols (None = all passing filters)
        min_dollar_volume: Minimum dollar volume
        rebalance_freq: Rebalance frequency

    Example:
        ```python
        # Get strata from provider
        strata = provider.get_strata("sector")

        # L1 blockchains only
        universe = CryptoStrata(strata=strata, include=["L1"])

        # DeFi and L2, excluding meme coins
        universe = CryptoStrata(
            strata=strata,
            include=["DeFi", "L2"],
            exclude=["Meme"]
        )

        # Top 20 L1 tokens
        universe = CryptoStrata(strata=strata, include=["L1"], top_n=20)
        ```
    """

    def __init__(
        self,
        strata: dict,
        include: Optional[list] = None,
        exclude: Optional[list] = None,
        top_n: Optional[int] = None,
        min_dollar_volume: float = 1_000_000,
        min_bars: int = 20,
        lookback: int = 20,
        rebalance_freq: str = "1d",
    ):
        inc_str = ",".join(include) if include else "All"
        exc_str = f"-{','.join(exclude)}" if exclude else ""
        n_str = f"_Top{top_n}" if top_n else ""
        name = f"Crypto_{inc_str}{exc_str}{n_str}"

        super().__init__(
            filters=[
                LiquidityFilter(min_dollar_volume=min_dollar_volume, lookback=lookback),
                StrataFilter(strata=strata, include=include, exclude=exclude),
                DataAvailabilityFilter(min_bars=min_bars),
            ],
            n=top_n,
            rebalance_freq=rebalance_freq,
            name=name,
        )


# ============================================================================
# Convenience aliases with common configurations
# ============================================================================

# Liquidity-based
CryptoTop10 = lambda **kw: CryptoLiquid(top_n=10, **kw)
CryptoTop20 = lambda **kw: CryptoLiquid(top_n=20, **kw)
CryptoTop30 = lambda **kw: CryptoLiquid(top_n=30, **kw)
CryptoTop50 = lambda **kw: CryptoLiquid(top_n=50, **kw)
CryptoTop100 = lambda **kw: CryptoLiquid(top_n=100, min_dollar_volume=500_000, **kw)

# Volatility-based
CryptoHighVol = lambda **kw: CryptoVolatility(min_vol=0.5, max_vol=3.0, **kw)
CryptoLowVol = lambda **kw: CryptoVolatility(min_vol=0.1, max_vol=0.8, **kw)
CryptoMidVol = lambda **kw: CryptoVolatility(min_vol=0.3, max_vol=1.0, **kw)


def CryptoL1Only(strata: dict, top_n: Optional[int] = None, **kwargs):
    """Layer 1 blockchain tokens only.

    Args:
        strata: Dict mapping symbol -> category (must have "L1" values)
        top_n: Number of top symbols (None = all)
        **kwargs: Additional arguments for CryptoStrata

    Example:
        ```python
        strata = provider.get_strata("layer")
        universe = CryptoL1Only(strata=strata, top_n=20)
        ```
    """
    return CryptoStrata(strata=strata, include=["L1"], top_n=top_n, **kwargs)


def CryptoDeFiOnly(strata: dict, top_n: Optional[int] = None, **kwargs):
    """DeFi tokens only.

    Args:
        strata: Dict mapping symbol -> category (must have "DeFi" values)
        top_n: Number of top symbols (None = all)
        **kwargs: Additional arguments for CryptoStrata
    """
    return CryptoStrata(
        strata=strata,
        include=["DeFi"],
        top_n=top_n,
        min_dollar_volume=500_000,
        **kwargs
    )


def CryptoL2Only(strata: dict, top_n: Optional[int] = None, **kwargs):
    """Layer 2 tokens only."""
    return CryptoStrata(strata=strata, include=["L2"], top_n=top_n, **kwargs)


def CryptoExcludeMeme(strata: dict, top_n: int = 50, **kwargs):
    """Liquid crypto excluding meme coins."""
    return CryptoStrata(strata=strata, exclude=["Meme"], top_n=top_n, **kwargs)


# Legacy aliases for backward compatibility
CryptoLiquid100 = lambda **kw: CryptoLiquid(top_n=100, min_dollar_volume=500_000, **kw)
CryptoHighVolatility = CryptoHighVol
CryptoLowVolatility = CryptoLowVol
