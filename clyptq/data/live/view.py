"""Live data view from rolling buffer."""

from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from clyptq.data.live.buffer import RollingPriceBuffer


class LiveDataView:
    """DataView interface for live trading with rolling buffer."""

    def __init__(self, buffer: RollingPriceBuffer, timestamp: datetime):
        self.buffer = buffer
        self._timestamp = timestamp

    @property
    def symbols(self) -> List[str]:
        return self.buffer.get_available_symbols()

    @property
    def timestamp(self) -> datetime:
        return self._timestamp

    def close(self, symbol: str, lookback: int = 100) -> Optional[np.ndarray]:
        return self.buffer.get_close_prices(symbol, lookback)

    def ohlcv(self, symbol: str, lookback: int = 100) -> Optional[pd.DataFrame]:
        return self.buffer.get_ohlcv(symbol, lookback)

    def returns(self, symbol: str, lookback: int = 100) -> Optional[np.ndarray]:
        prices = self.close(symbol, lookback + 1)
        if prices is None or len(prices) < 2:
            return None
        return np.diff(prices) / prices[:-1]

    def current_price(self, symbol: str) -> Optional[float]:
        prices = self.buffer.get_close_prices(symbol, 1)
        if prices is None:
            return None
        return float(prices[-1])

    def current_prices(self) -> Dict[str, float]:
        prices = {}
        for symbol in self.symbols:
            price = self.current_price(symbol)
            if price is not None:
                prices[symbol] = price
        return prices
