"""Base DataCollector abstraction.

DataCollector handles both historical and live data collection:
- collect_historical(): Bulk download for backtesting
- subscribe(): Real-time streaming for live trading

Each data type (OHLCV, Funding, Onchain) implements ONE collector
that supports BOTH modes.

Design:
- Collector is stateless - it just fetches data
- Data is passed to DataStore for storage
- DataFeed owns the collector and coordinates with Store
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import pandas as pd


class DataCollector(ABC):
    """Abstract data collector.

    Collectors fetch data from external sources.
    Same collector handles both historical and live modes.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Collector name (e.g., 'ccxt', 'glassnode')."""
        pass

    @property
    @abstractmethod
    def supported_fields(self) -> List[str]:
        """List of supported fields (e.g., ['open', 'high', 'low', 'close', 'volume'])."""
        pass

    @abstractmethod
    def collect_historical(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframe: str = "1h",
        **kwargs,
    ) -> Dict[str, pd.DataFrame]:
        """Collect historical data for symbols.

        Args:
            symbols: List of symbols to collect
            start: Start datetime
            end: End datetime
            timeframe: Data timeframe (e.g., "1h", "1d")
            **kwargs: Additional collector-specific options

        Returns:
            Dict mapping symbol -> DataFrame with collected data
            DataFrame should have datetime index and field columns
        """
        pass

    @abstractmethod
    def subscribe(
        self,
        symbols: List[str],
        on_data: Callable[[str, Dict[str, Any]], None],
        timeframe: str = "1h",
        **kwargs,
    ) -> "Subscription":
        """Subscribe to live data stream.

        Args:
            symbols: List of symbols to subscribe
            on_data: Callback for new data (symbol, data_dict)
            timeframe: Data timeframe
            **kwargs: Additional options

        Returns:
            Subscription handle for managing the subscription
        """
        pass

    def validate_symbols(self, symbols: List[str]) -> List[str]:
        """Validate and normalize symbols.

        Override in subclass for exchange-specific validation.

        Args:
            symbols: Raw symbol list

        Returns:
            Validated/normalized symbols
        """
        return symbols


class Subscription:
    """Handle for managing live data subscription."""

    def __init__(
        self,
        collector: DataCollector,
        symbols: List[str],
        cancel_fn: Optional[Callable[[], None]] = None,
    ):
        """Initialize subscription.

        Args:
            collector: Parent collector
            symbols: Subscribed symbols
            cancel_fn: Function to call on unsubscribe
        """
        self.collector = collector
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
