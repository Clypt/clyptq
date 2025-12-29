"""Rolling price buffer for live trading."""

from collections import deque
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd


class RollingPriceBuffer:
    """Stores recent price data for live factor computation."""

    def __init__(self, max_periods: int = 100):
        self.max_periods = max_periods
        self.prices: Dict[str, deque] = {}
        self.timestamps: deque = deque(maxlen=max_periods)

    def update(self, timestamp: datetime, prices: Dict[str, float]) -> None:
        """Add new price data."""
        self.timestamps.append(timestamp)

        for symbol, price in prices.items():
            if symbol not in self.prices:
                self.prices[symbol] = deque(maxlen=self.max_periods)
            self.prices[symbol].append(price)

    def get_close_prices(self, symbol: str, lookback: int) -> Optional[np.ndarray]:
        """Get recent close prices for a symbol."""
        if symbol not in self.prices:
            return None

        prices_list = list(self.prices[symbol])
        if len(prices_list) < lookback:
            return None

        return np.array(prices_list[-lookback:])

    def get_ohlcv(self, symbol: str, lookback: int) -> Optional[pd.DataFrame]:
        """Get OHLCV dataframe. For live trading, close = high = low = open."""
        close_prices = self.get_close_prices(symbol, lookback)
        if close_prices is None:
            return None

        timestamps_list = list(self.timestamps)[-lookback:]

        return pd.DataFrame({
            'timestamp': timestamps_list,
            'open': close_prices,
            'high': close_prices,
            'low': close_prices,
            'close': close_prices,
            'volume': np.zeros(len(close_prices)),
        })

    def get_available_symbols(self) -> list:
        """Get symbols with enough data."""
        return [s for s in self.prices.keys() if len(self.prices[s]) >= 2]

    def has_sufficient_data(self, symbol: str, min_periods: int) -> bool:
        """Check if symbol has enough data."""
        if symbol not in self.prices:
            return False
        return len(self.prices[symbol]) >= min_periods

    def clear(self) -> None:
        """Clear all data."""
        self.prices.clear()
        self.timestamps.clear()
