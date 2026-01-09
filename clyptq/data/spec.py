"""
Data specification classes.

DataSpec defines what data a strategy needs (not where to get it).
The runner determines the source based on mode (backtest vs live).

Example:
    ```python
    class MyStrategy(Strategy):
        data = {
            "ohlcv": OHLCVSpec(
                exchange="gateio",
                market_type="spot",
                timeframe="1d",
            ),
        }
    ```

Data path resolution (internal):
    Backtest: {DATA_ROOT}/{market_type}/{exchange}/{timeframe}/
    Example: data/spot/gateio/1d/BTCUSDT.parquet

    Live: Exchange API via CCXT
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union

# Supported exchanges
Exchange = Literal["gateio", "binance", "okx", "bybit", "upbit"]
MarketType = Literal["spot", "futures", "margin"]


@dataclass
class OHLCVSpec:
    """OHLCV data specification.

    Attributes:
        exchange: Exchange name (gateio, binance, okx, bybit, upbit)
        market_type: Market type (spot, futures, margin)
        timeframe: Data timeframe ("1m", "1h", "1d", etc.)
        fields: Which fields to load (default: all OHLCV)

    Example:
        ```python
        # Strategy with gateio spot data
        class MyStrategy(Strategy):
            data = {
                "ohlcv": OHLCVSpec(
                    exchange="gateio",
                    market_type="spot",
                    timeframe="1d",
                ),
            }
        ```
    """
    exchange: Exchange = "gateio"
    market_type: MarketType = "spot"
    timeframe: str = "1d"
    fields: List[str] = field(default_factory=lambda: ["open", "high", "low", "close", "volume"])

    def __post_init__(self):
        valid_fields = {"open", "high", "low", "close", "volume"}
        for f in self.fields:
            if f not in valid_fields:
                raise ValueError(f"Invalid OHLCV field: {f}")


@dataclass
class OrderBookSpec:
    """Order book data specification.

    Attributes:
        timeframe: Data timeframe (must be >= OHLCV timeframe)
        depth: Order book depth (number of levels)
        universe: Which universe this applies to (for multi-asset)
    """
    timeframe: str = "1m"
    depth: int = 10
    universe: Optional[str] = None  # None = all, "crypto" = crypto only

    @property
    def fields(self) -> List[str]:
        """Fields provided by this spec."""
        return [
            "bid_price", "bid_volume",
            "ask_price", "ask_volume",
            "mid_price", "spread",
        ]


@dataclass
class FundingSpec:
    """Funding rate data specification (crypto perpetuals).

    Funding rates are typically updated every 8 hours.

    Attributes:
        timeframe: Data timeframe (must be >= OHLCV timeframe)
        universe: Which universe ("crypto" for perpetuals)
    """
    timeframe: str = "8h"
    universe: str = "crypto"

    @property
    def fields(self) -> List[str]:
        return ["funding_rate", "next_funding_time"]


@dataclass
class TradesSpec:
    """Trade/tick data specification.

    Attributes:
        timeframe: Data timeframe (aggregation frequency)
        universe: Which universe this applies to
    """
    timeframe: str = "1m"
    universe: Optional[str] = None

    @property
    def fields(self) -> List[str]:
        return ["price", "volume", "side", "timestamp"]


# Type alias for any data spec
DataSpecType = Union[OHLCVSpec, OrderBookSpec, FundingSpec, TradesSpec]


@dataclass
class DataRequirements:
    """Complete data requirements for a strategy.

    This is built from Strategy.data dict automatically.

    Attributes:
        specs: Dict of {name: DataSpec}
        rebalance_freq: Rebalancing frequency
        warmup_periods: Required warmup bars
    """
    specs: dict = field(default_factory=dict)
    rebalance_freq: str = "1d"
    warmup_periods: int = 0

    @classmethod
    def from_strategy(cls, strategy: "Strategy") -> "DataRequirements":
        """Build requirements from strategy definition."""
        specs = getattr(strategy, "data", {"ohlcv": OHLCVSpec()})
        rebalance_freq = getattr(strategy, "rebalance_freq", "1d")
        warmup_periods = strategy.warmup_periods() if hasattr(strategy, "warmup_periods") else 0

        return cls(
            specs=specs,
            rebalance_freq=rebalance_freq,
            warmup_periods=warmup_periods,
        )

    @property
    def timeframes(self) -> List[str]:
        """Get all unique timeframes."""
        tfs = set()
        for spec in self.specs.values():
            if hasattr(spec, "timeframe"):
                tfs.add(spec.timeframe)
        return list(tfs)

    @property
    def needs_orderbook(self) -> bool:
        """Check if orderbook data is needed."""
        return any(isinstance(s, OrderBookSpec) for s in self.specs.values())

    @property
    def needs_funding(self) -> bool:
        """Check if funding data is needed."""
        return any(isinstance(s, FundingSpec) for s in self.specs.values())
