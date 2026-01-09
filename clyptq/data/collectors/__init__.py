"""Data collectors for historical and live data.

Collectors handle data fetching from external sources:
- Historical: Bulk download for backtesting
- Live: Real-time streaming via websockets

Each data type (OHLCV, Funding, Onchain) has ONE collector
that supports BOTH historical and live modes.
"""

from clyptq.data.collectors.base import DataCollector

__all__ = [
    "DataCollector",
]

# Optional: ccxt collector (requires ccxt package)
try:
    from clyptq.data.collectors.ccxt import CCXTCollector
    __all__.append("CCXTCollector")
except ImportError:
    pass
