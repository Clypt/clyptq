"""Data management layer.

Architecture:
- Provider: Main interface for data access (Buffer + Clock + Universe)
- Sources: Internal data loading abstraction (Parquet, Memory, Live)
- Collectors: Fetch data from external sources (historical + live)

Usage:
    ```python
    # Recommended: Use factory methods
    provider = DataProvider.from_parquet(
        path="data/crypto/",
        symbols=["BTC", "ETH"],
        rebalance_freq="1d",
    )
    provider.load(start=start, end=end)

    # Advanced: Custom source configuration
    from clyptq.data import ParquetSource
    source = ParquetSource(path="data/", timeframe="1h")
    provider = DataProvider(universe=my_universe, sources={"ohlcv": source})
    ```
"""

# Provider (main interface)
from clyptq.data.provider import DataProvider, SourceView

# Sources (internal, but importable for advanced use)
from clyptq.data.sources import (
    DataSource,
    OHLCVFields,
    ParquetSource,
    MemorySource,
    LiveSource,
)

# Collectors
from clyptq.data.collectors.base import DataCollector, Subscription

__all__ = [
    # Provider (main interface)
    "DataProvider",
    "SourceView",
    # Sources (for advanced use)
    "DataSource",
    "ParquetSource",
    "MemorySource",
    "LiveSource",
    # Collectors
    "DataCollector",
    "Subscription",
]

# Optional: ccxt collector
try:
    from clyptq.data.collectors.ccxt import CCXTCollector
    __all__.append("CCXTCollector")
except ImportError:
    pass
