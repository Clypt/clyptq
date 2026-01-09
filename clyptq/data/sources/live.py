"""Live streaming DataSource.

LiveSource wraps a DataCollector for real-time data streaming.
Maintains an internal buffer for lookback operations.
"""

from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from clyptq.data.sources.base import DataSource, OHLCVFields, Subscription

# Import collector types
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from clyptq.data.collectors.base import DataCollector


class LiveSource(DataSource):
    """Live streaming data source.

    Wraps a DataCollector for real-time data with internal buffering.

    Example:
        ```python
        collector = CCXTCollector("binance")
        source = LiveSource(collector, buffer_size=500)

        # Load historical data for warmup
        warmup_data = source.load(symbols, start, end)

        # Subscribe to live updates
        subscription = source.subscribe(
            symbols=["BTC", "ETH"],
            on_data=lambda sym, data: print(f"{sym}: {data}"),
        )

        # Get current buffer
        current_data = source.get_buffer()
        ```
    """

    def __init__(
        self,
        collector: "DataCollector",
        name: Optional[str] = None,
        timeframe: str = "1h",
        buffer_size: int = 500,
    ):
        """Initialize LiveSource.

        Args:
            collector: DataCollector for fetching data
            name: Source name
            timeframe: Data timeframe
            buffer_size: Maximum bars to keep in buffer per symbol
        """
        super().__init__(name=name, timeframe=timeframe)
        self.collector = collector
        self.buffer_size = buffer_size

        # Internal buffer: symbol -> DataFrame
        self._buffer: Dict[str, pd.DataFrame] = {}
        self._subscription: Optional[Subscription] = None

    @property
    def fields(self) -> List[str]:
        """Available fields from collector."""
        return self.collector.supported_fields

    def load(
        self,
        symbols: List[str],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        timeframe: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Load historical data via collector.

        Also populates the internal buffer for live operations.

        Args:
            symbols: Symbols to load
            start: Start datetime
            end: End datetime
            timeframe: Target timeframe

        Returns:
            Dict[field, DataFrame (T x N)]
        """
        tf = timeframe or self.timeframe

        # Fetch historical data via collector
        raw_data = self.collector.collect_historical(
            symbols=symbols,
            start=start or datetime.now() - timedelta(days=365),
            end=end or datetime.now(),
            timeframe=tf,
        )

        # Store in buffer
        for symbol, df in raw_data.items():
            self._add_to_buffer(symbol, df)

        # Convert to wide format
        return self._to_wide_format(symbols)

    def subscribe(
        self,
        symbols: List[str],
        on_data: Callable[[str, Dict[str, Any]], None],
        timeframe: Optional[str] = None,
    ) -> Subscription:
        """Subscribe to live data stream.

        Args:
            symbols: Symbols to subscribe
            on_data: Callback (symbol, data_dict)
            timeframe: Target timeframe

        Returns:
            Subscription handle
        """
        tf = timeframe or self.timeframe

        # Wrap callback to update buffer
        def wrapped_callback(symbol: str, data: Dict[str, Any]) -> None:
            self._update_buffer(symbol, data)
            on_data(symbol, data)

        # Subscribe via collector
        collector_sub = self.collector.subscribe(
            symbols=symbols,
            on_data=wrapped_callback,
            timeframe=tf,
        )

        # Wrap in our Subscription
        self._subscription = Subscription(
            source=self,
            symbols=symbols,
            cancel_fn=collector_sub.unsubscribe,
        )

        return self._subscription

    def get_buffer(self, symbols: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """Get current buffer data in wide format.

        Args:
            symbols: Subset of symbols (None = all)

        Returns:
            Dict[field, DataFrame (T x N)]
        """
        target_symbols = symbols or list(self._buffer.keys())
        return self._to_wide_format(target_symbols)

    def _add_to_buffer(self, symbol: str, df: pd.DataFrame) -> None:
        """Add historical data to buffer."""
        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
            else:
                raise ValueError("DataFrame must have DatetimeIndex or 'timestamp' column")

        df = df.sort_index()

        if symbol in self._buffer:
            # Merge with existing
            self._buffer[symbol] = pd.concat([self._buffer[symbol], df])
            self._buffer[symbol] = self._buffer[symbol][
                ~self._buffer[symbol].index.duplicated(keep="last")
            ]
        else:
            self._buffer[symbol] = df

        # Trim to buffer size
        if len(self._buffer[symbol]) > self.buffer_size:
            self._buffer[symbol] = self._buffer[symbol].iloc[-self.buffer_size:]

    def _update_buffer(self, symbol: str, data: Dict[str, Any]) -> None:
        """Update buffer with new tick data."""
        timestamp = data.get("timestamp", datetime.now())
        if isinstance(timestamp, (int, float)):
            timestamp = datetime.fromtimestamp(timestamp / 1000)

        new_row = pd.DataFrame(
            [{
                "open": data.get("open", data.get("price", 0)),
                "high": data.get("high", data.get("price", 0)),
                "low": data.get("low", data.get("price", 0)),
                "close": data.get("close", data.get("price", 0)),
                "volume": data.get("volume", 0),
            }],
            index=[timestamp],
        )

        if symbol in self._buffer:
            self._buffer[symbol] = pd.concat([self._buffer[symbol], new_row])
            self._buffer[symbol] = self._buffer[symbol][
                ~self._buffer[symbol].index.duplicated(keep="last")
            ]
            # Trim
            if len(self._buffer[symbol]) > self.buffer_size:
                self._buffer[symbol] = self._buffer[symbol].iloc[-self.buffer_size:]
        else:
            self._buffer[symbol] = new_row

    def _to_wide_format(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Convert buffer to wide format."""
        result: Dict[str, pd.DataFrame] = {}

        for field in OHLCVFields.ALL:
            series_dict = {}
            for symbol in symbols:
                if symbol in self._buffer and field in self._buffer[symbol].columns:
                    series_dict[symbol] = self._buffer[symbol][field]

            if series_dict:
                result[field] = pd.DataFrame(series_dict)

        return result

    def available_symbols(self) -> List[str]:
        """Get symbols currently in buffer."""
        return sorted(self._buffer.keys())

    def clear_buffer(self) -> None:
        """Clear the internal buffer."""
        self._buffer.clear()
