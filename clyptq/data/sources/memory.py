"""In-memory DataSource.

MemorySource stores data in memory as Dict[symbol, DataFrame].
Used for:
- Testing
- Small datasets
- Data already loaded from external sources

This replaces the old DataStore class functionality.
"""

from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from clyptq.data.sources.base import DataSource, OHLCVFields


class MemorySource(DataSource):
    """In-memory data source.

    Stores OHLCV data in memory. Data is added via add_symbol() method.

    Example:
        ```python
        source = MemorySource()
        source.add_symbol("BTC", btc_df)
        source.add_symbol("ETH", eth_df)

        data = source.load(symbols=["BTC", "ETH"], start=start, end=end)
        close = data["close"]  # DataFrame (T x N)
        ```
    """

    def __init__(
        self,
        name: Optional[str] = None,
        timeframe: str = "1d",
        exchange: Optional[str] = None,
    ):
        """Initialize MemorySource.

        Args:
            name: Source name
            timeframe: Data timeframe
            exchange: Exchange identifier (e.g., "binance", "mock")
        """
        super().__init__(name=name, timeframe=timeframe, exchange=exchange)
        self._data: Dict[str, pd.DataFrame] = {}

    @property
    def fields(self) -> List[str]:
        """OHLCV fields."""
        return OHLCVFields.ALL

    def add_symbol(
        self,
        symbol: str,
        df: pd.DataFrame,
    ) -> None:
        """Add data for a symbol.

        Args:
            symbol: Symbol identifier
            df: DataFrame with OHLCV columns and DatetimeIndex

        Raises:
            ValueError: If DataFrame format is invalid
        """
        required_cols = OHLCVFields.ALL
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must have columns: {required_cols}")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex")

        # Sort and deduplicate
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]

        self._data[symbol] = df

    def load(
        self,
        symbols: List[str],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        timeframe: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Load data for symbols.

        Args:
            symbols: Symbols to load
            start: Start datetime
            end: End datetime
            timeframe: Target timeframe (resampling if different from source)

        Returns:
            Dict[field, DataFrame (T x N)]
        """
        target_tf = timeframe or self.timeframe
        needs_resample = target_tf != self.timeframe

        result: Dict[str, pd.DataFrame] = {}

        for field in self.fields:
            series_dict = {}

            for symbol in symbols:
                if symbol not in self._data:
                    continue

                df = self._data[symbol]

                # Apply date filters
                if start is not None:
                    df = df[df.index >= start]
                if end is not None:
                    df = df[df.index <= end]

                if len(df) == 0:
                    continue

                # Resample if needed
                if needs_resample:
                    df = self._resample(df, target_tf)

                if len(df) > 0:
                    series_dict[symbol] = df[field]

            if series_dict:
                result[field] = pd.DataFrame(series_dict)

        return result

    def _resample(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample OHLCV data to target timeframe."""
        rule = self._timeframe_to_rule(timeframe)

        return df.resample(rule).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

    def _timeframe_to_rule(self, timeframe: str) -> str:
        """Convert timeframe to pandas resample rule."""
        if timeframe.endswith("m"):
            return f"{timeframe[:-1]}min"
        elif timeframe.endswith("h"):
            return f"{timeframe[:-1]}h"
        elif timeframe.endswith("d"):
            return f"{timeframe[:-1]}D"
        elif timeframe.endswith("w"):
            return f"{timeframe[:-1]}W"
        else:
            raise ValueError(f"Unknown timeframe: {timeframe}")

    def available_symbols(self) -> List[str]:
        """Get list of available symbols."""
        return sorted(self._data.keys())

    def has_symbol(self, symbol: str) -> bool:
        """Check if symbol exists."""
        return symbol in self._data

    def remove_symbol(self, symbol: str) -> None:
        """Remove symbol from source."""
        if symbol in self._data:
            del self._data[symbol]

    def clear(self) -> None:
        """Clear all data."""
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, symbol: str) -> bool:
        return symbol in self._data
