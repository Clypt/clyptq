"""Real-time data store with rolling window for live/paper trading."""

from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd

from clyptq.data.stores.store import DataView


class LiveDataStore:
    """Data store for live/paper trading with rolling window."""

    def __init__(self, lookback_days: int = 90):
        self.lookback_days = lookback_days
        self.data: Dict[str, pd.DataFrame] = {}
        self._last_update: Dict[str, datetime] = {}

    def add_historical(self, symbol: str, df: pd.DataFrame) -> None:
        """Add historical OHLCV data for warmup.

        Args:
            symbol: Trading symbol
            df: DataFrame with columns [timestamp, open, high, low, close, volume]
        """
        if "timestamp" not in df.columns:
            raise ValueError("DataFrame must have 'timestamp' column")

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()

        if len(df) > 0:
            cutoff = df.index[-1] - timedelta(days=self.lookback_days - 1)
            df = df[df.index >= cutoff]

        self.data[symbol] = df
        if len(df) > 0:
            self._last_update[symbol] = df.index[-1]

    def update(self, timestamp: datetime, prices: Dict[str, float]) -> None:
        """Update with new prices (creates daily bar).

        Args:
            timestamp: Current timestamp
            prices: {symbol: price}
        """
        cutoff = timestamp - timedelta(days=self.lookback_days - 1)

        for symbol, price in prices.items():
            if symbol not in self.data:
                self.data[symbol] = pd.DataFrame(
                    columns=["open", "high", "low", "close", "volume"]
                )
                self.data[symbol].index.name = "timestamp"

            new_bar = pd.DataFrame(
                {
                    "open": [price],
                    "high": [price],
                    "low": [price],
                    "close": [price],
                    "volume": [0.0],
                },
                index=[timestamp],
            )

            self.data[symbol] = pd.concat([self.data[symbol], new_bar])
            self.data[symbol] = self.data[symbol][self.data[symbol].index >= cutoff]
            self.data[symbol] = self.data[symbol][
                ~self.data[symbol].index.duplicated(keep="last")
            ]

            self._last_update[symbol] = timestamp

    def get_view(self, timestamp: datetime) -> DataView:
        """Get point-in-time data view.

        Args:
            timestamp: Timestamp for view

        Returns:
            DataView at timestamp
        """
        return DataView(self.data, timestamp)

    def available_symbols(self, timestamp: datetime, min_bars: int = 20) -> List[str]:
        """Get symbols with sufficient data.

        Args:
            timestamp: Current timestamp
            min_bars: Minimum required bars

        Returns:
            List of symbols with enough data
        """
        symbols = []
        for symbol, df in self.data.items():
            mask = df.index <= timestamp
            if mask.sum() >= min_bars:
                symbols.append(symbol)
        return symbols

    def has_symbol(self, symbol: str) -> bool:
        """Check if symbol exists."""
        return symbol in self.data

    def num_symbols(self) -> int:
        """Number of tracked symbols."""
        return len(self.data)

    def reset(self) -> None:
        """Clear all data."""
        self.data.clear()
        self._last_update.clear()
