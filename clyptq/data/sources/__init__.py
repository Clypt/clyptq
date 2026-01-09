"""Data sources - unified interface for data loading.

DataSource provides a single interface for:
- Backtest: Load from files (Parquet, CSV) or DB
- Live: Stream from WebSocket via Collector

All sources share the same interface:
    - fields: List of available fields
    - load(): Load historical data
    - subscribe(): Subscribe to live stream (optional)
"""

from clyptq.data.sources.base import DataSource, OHLCVFields
from clyptq.data.sources.parquet import ParquetSource
from clyptq.data.sources.memory import MemorySource
from clyptq.data.sources.live import LiveSource

__all__ = [
    "DataSource",
    "OHLCVFields",
    "ParquetSource",
    "MemorySource",
    "LiveSource",
]
