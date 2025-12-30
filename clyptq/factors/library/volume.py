from typing import Dict

from clyptq.core.base import Factor
from clyptq.factors.ops.time_series import ts_mean


class VolumeFactor(Factor):
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

            volume = df["volume"].values
            avg_volume = ts_mean(volume, self.lookback)
            if avg_volume == 0:
                continue

            recent_volume = volume[-1]
            scores[symbol] = recent_volume / avg_volume

        return scores


class DollarVolumeFactor(Factor):
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

            scores[symbol] = avg_dollar_volume

        return scores


class VolumeRatioFactor(Factor):
    def __init__(self, short_window: int = 5, long_window: int = 20):
        super().__init__()
        self.short_window = short_window
        self.long_window = long_window

    def compute(self, data) -> Dict[str, float]:
        scores = {}
        for symbol in data.symbols:
            try:
                df = data.ohlcv(symbol, self.long_window)
            except (KeyError, ValueError):
                continue

            volume = df["volume"].values
            short_avg = ts_mean(volume[-self.short_window :], self.short_window)
            long_avg = ts_mean(volume, self.long_window)

            if long_avg == 0:
                continue

            scores[symbol] = short_avg / long_avg

        return scores
