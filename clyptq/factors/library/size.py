import numpy as np
from typing import Dict

from clyptq.core.base import Factor
from clyptq.factors.ops.time_series import ts_mean


class DollarVolumeSizeFactor(Factor):
    def __init__(self, lookback: int = 20):
        super().__init__()
        self.lookback = lookback

    def compute(self, data) -> Dict[str, float]:
        scores = {}
        for symbol in data.symbols:
            try:
                df = data.ohlcv(symbol, self.lookback)
            except (KeyError, ValueError):
                continue

            close = df["close"].values
            volume = df["volume"].values
            dollar_volume = close * volume
            avg_dollar_volume = ts_mean(dollar_volume, self.lookback)

            scores[symbol] = np.log(avg_dollar_volume + 1)

        return scores
