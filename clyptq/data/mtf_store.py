"""
Multi-timeframe data store for strategies using multiple timeframes.

Stores OHLCV data at different resolutions (1h, 4h, 1d, 1w) with
automatic alignment and point-in-time consistency.
"""

from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from clyptq.data.store import DataStore


class MultiTimeframeStore:
    """
    Store multiple timeframes per symbol with alignment utilities.

    Supports common crypto timeframes:
    - 1h: Hourly bars
    - 4h: 4-hour bars
    - 1d: Daily bars
    - 1w: Weekly bars

    Maintains point-in-time consistency across timeframes.
    """

    VALID_TIMEFRAMES = ["1h", "4h", "1d", "1w"]

    def __init__(self):
        # timeframe -> DataStore mapping
        self.stores: Dict[str, DataStore] = {}
        self._initialize_stores()

    def _initialize_stores(self) -> None:
        """Initialize DataStore for each timeframe."""
        for tf in self.VALID_TIMEFRAMES:
            self.stores[tf] = DataStore()

    def add_ohlcv(
        self, symbol: str, data: pd.DataFrame, timeframe: str = "1d"
    ) -> None:
        """
        Add OHLCV data for a symbol at specific timeframe.

        Args:
            symbol: Trading symbol (e.g., "BTC/USDT")
            data: OHLCV DataFrame with DatetimeIndex
            timeframe: Timeframe string ("1h", "4h", "1d", "1w")

        Raises:
            ValueError: If timeframe is invalid
        """
        if timeframe not in self.VALID_TIMEFRAMES:
            raise ValueError(
                f"Invalid timeframe '{timeframe}'. Must be one of {self.VALID_TIMEFRAMES}"
            )

        self.stores[timeframe].add_ohlcv(symbol, data)

    def get_store(self, timeframe: str) -> DataStore:
        """
        Get DataStore for specific timeframe.

        Args:
            timeframe: Timeframe string

        Returns:
            DataStore for that timeframe

        Raises:
            ValueError: If timeframe is invalid
        """
        if timeframe not in self.VALID_TIMEFRAMES:
            raise ValueError(
                f"Invalid timeframe '{timeframe}'. Must be one of {self.VALID_TIMEFRAMES}"
            )

        return self.stores[timeframe]

    def available_timeframes(self, symbol: str) -> List[str]:
        """
        Get list of available timeframes for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            List of timeframe strings where data exists
        """
        available = []
        for tf in self.VALID_TIMEFRAMES:
            if symbol in self.stores[tf]._data:
                available.append(tf)
        return available

    def resample_to_timeframe(
        self,
        symbol: str,
        source_tf: str,
        target_tf: str,
    ) -> Optional[pd.DataFrame]:
        """
        Resample data from source to target timeframe.

        Only upsampling is supported (e.g., 1h -> 1d, 1d -> 1w).
        Downsampling (e.g., 1d -> 1h) returns None.

        Args:
            symbol: Trading symbol
            source_tf: Source timeframe
            target_tf: Target timeframe

        Returns:
            Resampled DataFrame or None if not possible

        Raises:
            ValueError: If timeframes are invalid
        """
        if source_tf not in self.VALID_TIMEFRAMES:
            raise ValueError(f"Invalid source timeframe: {source_tf}")
        if target_tf not in self.VALID_TIMEFRAMES:
            raise ValueError(f"Invalid target timeframe: {target_tf}")

        # Check if data exists
        if symbol not in self.stores[source_tf]._data:
            return None

        # Get source data
        source_data = self.stores[source_tf]._data[symbol]

        # Define resampling rules
        resample_map = {
            "1h": {"4h": "4h", "1d": "D", "1w": "W"},
            "4h": {"1d": "D", "1w": "W"},
            "1d": {"1w": "W"},
        }

        # Check if upsampling is possible
        if source_tf not in resample_map:
            return None
        if target_tf not in resample_map[source_tf]:
            return None

        # Resample
        rule = resample_map[source_tf][target_tf]
        resampled = source_data.resample(rule).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )

        return resampled.dropna()

    def align_timestamps(
        self,
        timestamp: datetime,
        timeframes: List[str],
    ) -> Dict[str, datetime]:
        """
        Align timestamp to each timeframe's bar.

        For each timeframe, returns the most recent bar timestamp
        that is <= the input timestamp (point-in-time).

        Args:
            timestamp: Current timestamp
            timeframes: List of timeframes to align

        Returns:
            Dict mapping timeframe -> aligned timestamp
        """
        aligned = {}

        for tf in timeframes:
            if tf not in self.VALID_TIMEFRAMES:
                continue

            # Get frequency for resampling
            freq_map = {"1h": "h", "4h": "4h", "1d": "D", "1w": "W"}
            freq = freq_map[tf]

            # Floor to timeframe boundary
            if tf == "1h":
                aligned_ts = timestamp.replace(minute=0, second=0, microsecond=0)
            elif tf == "4h":
                hour = (timestamp.hour // 4) * 4
                aligned_ts = timestamp.replace(hour=hour, minute=0, second=0, microsecond=0)
            elif tf == "1d":
                aligned_ts = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            elif tf == "1w":
                # Floor to Monday
                days_since_monday = timestamp.weekday()
                aligned_ts = timestamp - pd.Timedelta(days=days_since_monday)
                aligned_ts = aligned_ts.replace(hour=0, minute=0, second=0, microsecond=0)

            aligned[tf] = aligned_ts

        return aligned

    def get_bar_at_timestamp(
        self,
        symbol: str,
        timeframe: str,
        timestamp: datetime,
    ) -> Optional[pd.Series]:
        """
        Get OHLCV bar for symbol at specific timeframe and timestamp.

        Returns the most recent bar <= timestamp (point-in-time).

        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            timestamp: Query timestamp

        Returns:
            OHLCV bar as Series or None if not available
        """
        if timeframe not in self.VALID_TIMEFRAMES:
            raise ValueError(f"Invalid timeframe: {timeframe}")

        store = self.stores[timeframe]
        if symbol not in store._data:
            return None

        data = store._data[symbol]

        # Get most recent bar <= timestamp
        available = data[data.index <= timestamp]
        if available.empty:
            return None

        return available.iloc[-1]

    def has_sufficient_data(
        self,
        symbol: str,
        timeframe: str,
        timestamp: datetime,
        lookback: int,
    ) -> bool:
        """
        Check if sufficient historical data exists for computation.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            timestamp: Current timestamp
            lookback: Required number of bars

        Returns:
            True if sufficient data exists
        """
        if timeframe not in self.VALID_TIMEFRAMES:
            return False

        store = self.stores[timeframe]
        if symbol not in store._data:
            return False

        data = store._data[symbol]
        available = data[data.index <= timestamp]

        return len(available) >= lookback
