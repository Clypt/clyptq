"""Base DataSource abstraction.

DataSource is the unified interface for loading data, regardless of:
- Storage backend (Parquet, DB, Memory)
- Mode (Backtest vs Live)

Design:
- load() returns Dict[field, DataFrame] in wide format (T x N)
- subscribe() is optional, only for live sources
- Resampling is handled by the source if needed
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import pandas as pd


class OHLCVFields:
    """Standard OHLCV field names."""

    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    VOLUME = "volume"

    ALL = ["open", "high", "low", "close", "volume"]


class DataSource(ABC):
    """Abstract base class for data sources.

    DataSource provides a unified interface for loading data from any backend.
    Both historical (backtest) and live modes use the same interface.

    Exchange Support:
        Each source can optionally specify which exchange the data comes from.
        This is useful when:
        - Same symbol has different prices on different exchanges
        - You want to combine data from multiple exchanges
        - Trading execution needs to know the data source

    Example:
        ```python
        # Backtest with Parquet files (single exchange)
        source = ParquetSource(
            path="data/binance/",
            exchange="binance"
        )
        data = source.load(
            symbols=["BTC", "ETH"],
            start=datetime(2023, 1, 1),
            end=datetime(2024, 1, 1),
        )
        close_df = data["close"]  # DataFrame (T x N)

        # Multiple exchanges
        provider = DataProvider(
            sources={
                "binance": ParquetSource(path="data/binance/", exchange="binance"),
                "upbit": ParquetSource(path="data/upbit/", exchange="upbit"),
            }
        )

        # Live with WebSocket
        source = LiveSource(collector=BinanceCollector(), exchange="binance")
        source.subscribe(symbols=["BTC", "ETH"], on_data=callback)
        ```
    """

    def __init__(
        self,
        name: Optional[str] = None,
        timeframe: str = "1d",
        exchange: Optional[str] = None,
    ):
        """Initialize DataSource.

        Args:
            name: Source name (defaults to class name)
            timeframe: Default timeframe for this source
            exchange: Exchange identifier (e.g., "binance", "upbit", "okx")
        """
        self.name = name or self.__class__.__name__
        self.timeframe = timeframe
        self.exchange = exchange

    @property
    @abstractmethod
    def fields(self) -> List[str]:
        """Available fields from this source.

        Returns:
            List of field names (e.g., ["open", "high", "low", "close", "volume"])
        """
        pass

    @abstractmethod
    def load(
        self,
        symbols: List[str],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        timeframe: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Load data for symbols.

        Args:
            symbols: List of symbols to load
            start: Start datetime (None = earliest available)
            end: End datetime (None = latest available)
            timeframe: Target timeframe (None = use source default)

        Returns:
            Dict mapping field name to DataFrame (T x N)
            - Index: DatetimeIndex (timestamps)
            - Columns: Symbol names
        """
        pass

    def subscribe(
        self,
        symbols: List[str],
        on_data: Callable[[str, Dict[str, Any]], None],
        timeframe: Optional[str] = None,
    ) -> "Subscription":
        """Subscribe to live data stream.

        Default implementation raises NotImplementedError.
        Override in live-capable sources.

        Args:
            symbols: Symbols to subscribe
            on_data: Callback (symbol, data_dict)
            timeframe: Target timeframe

        Returns:
            Subscription handle

        Raises:
            NotImplementedError: If source doesn't support live streaming
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support live streaming"
        )

    def available_symbols(self) -> List[str]:
        """Get list of available symbols.

        Default implementation returns empty list.
        Override in sources that can enumerate symbols.

        Returns:
            List of available symbol names
        """
        return []

    def __repr__(self) -> str:
        exchange_str = f", exchange={self.exchange}" if self.exchange else ""
        return f"{self.__class__.__name__}(name={self.name}, timeframe={self.timeframe}{exchange_str})"


class Subscription:
    """Handle for managing live data subscription."""

    def __init__(
        self,
        source: DataSource,
        symbols: List[str],
        cancel_fn: Optional[Callable[[], None]] = None,
    ):
        """Initialize subscription.

        Args:
            source: Parent source
            symbols: Subscribed symbols
            cancel_fn: Function to call on unsubscribe
        """
        self.source = source
        self.symbols = symbols
        self._cancel_fn = cancel_fn
        self._active = True

    @property
    def is_active(self) -> bool:
        """Check if subscription is active."""
        return self._active

    def unsubscribe(self) -> None:
        """Cancel the subscription."""
        if self._active:
            if self._cancel_fn:
                self._cancel_fn()
            self._active = False

    def __enter__(self) -> "Subscription":
        return self

    def __exit__(self, *args) -> None:
        self.unsubscribe()
