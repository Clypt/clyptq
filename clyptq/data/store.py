"""Data storage with look-ahead bias prevention."""

from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from clyptq.core.types import OHLCV, DataMetadata, DataRange


class DataView:
    """Time-point data view. No look-ahead."""

    def __init__(self, data: Dict[str, pd.DataFrame], timestamp: datetime):
        """Init at timestamp."""
        self._data = data
        self._timestamp = timestamp

    @property
    def symbols(self) -> List[str]:
        """Available symbols."""
        return list(self._data.keys())

    @property
    def timestamp(self) -> datetime:
        """Current timestamp."""
        return self._timestamp

    def close(self, symbol: str, lookback: int = 100) -> np.ndarray:
        """Get close prices."""
        if symbol not in self._data:
            raise KeyError(f"Symbol {symbol} not found")

        df = self._data[symbol]

        mask = df.index <= self._timestamp
        available = df.loc[mask]

        if len(available) < lookback:
            raise ValueError(
                f"Insufficient data for {symbol}: need {lookback}, have {len(available)}"
            )

        return available["close"].iloc[-lookback:].values

    def ohlcv(self, symbol: str, lookback: int = 100) -> pd.DataFrame:
        """Get OHLCV data."""
        if symbol not in self._data:
            raise KeyError(f"Symbol {symbol} not found")

        df = self._data[symbol]

        mask = df.index <= self._timestamp
        available = df.loc[mask]

        if len(available) < lookback:
            raise ValueError(
                f"Insufficient data for {symbol}: need {lookback}, have {len(available)}"
            )

        return available[["open", "high", "low", "close", "volume"]].iloc[-lookback:]

    def returns(self, symbol: str, lookback: int = 100) -> np.ndarray:
        """Simple returns."""
        prices = self.close(symbol, lookback + 1)
        return np.diff(prices) / prices[:-1]

    def log_returns(self, symbol: str, lookback: int = 100) -> np.ndarray:
        """Log returns."""
        prices = self.close(symbol, lookback + 1)
        return np.diff(np.log(prices))

    def current_price(self, symbol: str) -> Optional[float]:
        """Current close price."""
        try:
            return float(self.close(symbol, lookback=1)[0])
        except (KeyError, ValueError):
            return None

    def current_prices(self) -> Dict[str, float]:
        """All current prices."""
        prices = {}
        for symbol in self.symbols:
            price = self.current_price(symbol)
            if price is not None:
                prices[symbol] = price
        return prices

    def has_data(self, symbol: str, min_bars: int = 1) -> bool:
        """Check sufficient data."""
        if symbol not in self._data:
            return False

        df = self._data[symbol]
        mask = df.index <= self._timestamp
        return mask.sum() >= min_bars


class DataStore:
    """Data storage. No look-ahead bias."""

    def __init__(self):
        """Init empty store."""
        self._data: Dict[str, pd.DataFrame] = {}
        self._metadata: Dict[str, DataMetadata] = {}

    def add_ohlcv(
        self,
        symbol: str,
        df: pd.DataFrame,
        frequency: str = "1d",
        source: str = "unknown",
    ) -> None:
        """
        Add OHLCV data for a symbol.

        Args:
            symbol: Symbol identifier
            df: DataFrame with columns [open, high, low, close, volume] and DatetimeIndex
            frequency: Data frequency (e.g., "1m", "1h", "1d")
            source: Data source identifier

        Raises:
            ValueError: If DataFrame format is invalid
        """
        # Validate DataFrame
        required_cols = ["open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must have columns: {required_cols}")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex")

        # Sort by timestamp and remove duplicates
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]

        # Store data
        self._data[symbol] = df

        # Store metadata
        self._metadata[symbol] = DataMetadata(
            symbol=symbol,
            start_date=df.index.min().to_pydatetime(),
            end_date=df.index.max().to_pydatetime(),
            num_bars=len(df),
            frequency=frequency,
            source=source,
        )

    def add_ohlcv_list(self, symbol: str, ohlcv_list: List[OHLCV]) -> None:
        """
        Add OHLCV data from list of OHLCV objects.

        Args:
            symbol: Symbol identifier
            ohlcv_list: List of OHLCV objects
        """
        if not ohlcv_list:
            raise ValueError("OHLCV list cannot be empty")

        # Convert to DataFrame
        data = {
            "open": [o.open for o in ohlcv_list],
            "high": [o.high for o in ohlcv_list],
            "low": [o.low for o in ohlcv_list],
            "close": [o.close for o in ohlcv_list],
            "volume": [o.volume for o in ohlcv_list],
        }
        index = [o.timestamp for o in ohlcv_list]

        df = pd.DataFrame(data, index=pd.DatetimeIndex(index))
        self.add_ohlcv(symbol, df)

    def get_view(self, timestamp: datetime) -> DataView:
        """
        Get time-point view of data.

        Args:
            timestamp: Timestamp for the view

        Returns:
            DataView at the specified timestamp
        """
        return DataView(self._data, timestamp)

    def available_symbols(self, at: datetime) -> List[str]:
        """
        Get symbols that have data at the given timestamp.

        CRITICAL: Checks if symbol was listed AND not yet delisted.
        Prevents look-ahead bias by using only actual data availability.

        Args:
            at: Timestamp to check

        Returns:
            List of available symbols
        """
        available = []
        for symbol, df in self._data.items():
            # Symbol must be: already listed AND still has data at this point
            # Data availability naturally reflects delisting without look-ahead
            if df.index.min() <= at <= df.index.max():
                available.append(symbol)
        return available

    def get_top_symbols_by_volume(
        self, at: datetime, top_n: int, lookback_days: int = 7
    ) -> List[str]:
        """Top N by volume. Uses only past data to avoid bias."""
        available = self.available_symbols(at)

        if len(available) == 0:
            return []

        # Calculate average volume for each symbol
        symbol_volumes = []

        for symbol in available:
            df = self._data[symbol]

            # Get data up to and including 'at' timestamp
            mask = df.index <= at

            if not mask.any():
                continue

            # Get recent data (last lookback_days)
            recent_data = df[mask].tail(lookback_days)

            if len(recent_data) == 0:
                continue

            # Calculate average volume
            avg_volume = recent_data["volume"].mean()

            symbol_volumes.append((symbol, avg_volume))

        # Sort by volume descending
        symbol_volumes.sort(key=lambda x: x[1], reverse=True)

        # Return top N symbols
        top_symbols = [s[0] for s in symbol_volumes[:top_n]]

        return top_symbols

    def get_date_range(self, symbol: Optional[str] = None) -> DataRange:
        """
        Get date range for stored data.

        Args:
            symbol: If specified, get range for this symbol only

        Returns:
            DataRange with start and end dates

        Raises:
            ValueError: If no data available
        """
        if symbol:
            if symbol not in self._metadata:
                raise ValueError(f"No data for symbol: {symbol}")
            meta = self._metadata[symbol]
            return DataRange(start=meta.start_date, end=meta.end_date)

        if not self._metadata:
            raise ValueError("No data available")

        # Get overall range across all symbols
        start = min(m.start_date for m in self._metadata.values())
        end = max(m.end_date for m in self._metadata.values())
        return DataRange(start=start, end=end)

    def get_metadata(self, symbol: str) -> DataMetadata:
        """
        Get metadata for a symbol.

        Args:
            symbol: Symbol to query

        Returns:
            DataMetadata for the symbol

        Raises:
            KeyError: If symbol not found
        """
        if symbol not in self._metadata:
            raise KeyError(f"No data for symbol: {symbol}")
        return self._metadata[symbol]

    def symbols(self) -> List[str]:
        """Get list of all stored symbols."""
        return list(self._data.keys())

    def has_symbol(self, symbol: str) -> bool:
        """Check if symbol exists in store."""
        return symbol in self._data

    def remove_symbol(self, symbol: str) -> None:
        """Remove symbol from store."""
        if symbol in self._data:
            del self._data[symbol]
        if symbol in self._metadata:
            del self._metadata[symbol]

    def clear(self) -> None:
        """Remove all data from store."""
        self._data.clear()
        self._metadata.clear()

    def __len__(self) -> int:
        """Get number of symbols in store."""
        return len(self._data)

    def __contains__(self, symbol: str) -> bool:
        """Check if symbol in store."""
        return symbol in self._data

    def __repr__(self) -> str:
        """String representation."""
        return f"DataStore({len(self)} symbols)"
