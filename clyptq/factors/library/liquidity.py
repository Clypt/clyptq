import numpy as np
from typing import Dict

from clyptq.core.base import Factor
from clyptq.factors.ops.time_series import ts_mean


class AmihudFactor(Factor):
    def __init__(self, lookback: int = 20):
        super().__init__()
        self.lookback = lookback

    def compute(self, data) -> Dict[str, float]:
        scores = {}
        for symbol in data.symbols:
            try:
                df = data.ohlcv(symbol, self.lookback + 1)
            except (KeyError, ValueError):
                continue

            close = df["close"].values
            volume = df["volume"].values

            returns = np.diff(close) / close[:-1]
            dollar_volume = close[1:] * volume[1:]

            illiquidity = np.abs(returns) / (dollar_volume + 1e-10)
            avg_illiquidity = ts_mean(illiquidity, self.lookback)

            scores[symbol] = -avg_illiquidity

        return scores


class EffectiveSpreadFactor(Factor):
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

            high = df["high"].values
            low = df["low"].values
            close = df["close"].values

            spread = (high - low) / (close + 1e-10)
            avg_spread = ts_mean(spread, self.lookback)

            scores[symbol] = -avg_spread

        return scores


class VolatilityOfVolatilityFactor(Factor):
    def __init__(self, lookback: int = 20, vol_window: int = 5):
        super().__init__()
        self.lookback = lookback
        self.vol_window = vol_window

    def compute(self, data) -> Dict[str, float]:
        scores = {}
        for symbol in data.symbols:
            try:
                df = data.ohlcv(symbol, self.lookback + self.vol_window)
            except (KeyError, ValueError):
                continue

            close = df["close"].values
            returns = np.diff(close) / close[:-1]

            rolling_vol = []
            for i in range(len(returns) - self.vol_window + 1):
                window = returns[i : i + self.vol_window]
                rolling_vol.append(np.std(window))

            rolling_vol = np.array(rolling_vol)
            if len(rolling_vol) < self.lookback:
                continue

            vol_of_vol = np.std(rolling_vol[-self.lookback :])
            scores[symbol] = -vol_of_vol

        return scores
