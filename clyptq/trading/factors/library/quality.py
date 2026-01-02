import numpy as np
from typing import Dict

from clyptq.core.base import Factor
from clyptq.trading.factors.ops.time_series import ts_mean, ts_std


class VolumeStabilityFactor(Factor):
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

            mean_vol = ts_mean(volume, self.lookback)
            std_vol = ts_std(volume, self.lookback)

            if mean_vol < 1e-10:
                continue

            cv = std_vol / mean_vol
            stability = 1.0 / (cv + 1e-10)

            scores[symbol] = stability

        return scores


class PriceImpactFactor(Factor):
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

            returns = np.abs(np.diff(close) / close[:-1])
            log_volume = np.log(volume[1:] + 1.0)

            if len(returns) < self.lookback:
                continue

            impact = returns / (log_volume + 1e-10)
            avg_impact = ts_mean(impact, self.lookback)

            scores[symbol] = -avg_impact

        return scores


class MarketDepthProxyFactor(Factor):
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

            if len(returns) < self.lookback:
                continue

            volatility = ts_std(returns, self.lookback)

            if volatility < 1e-10:
                continue

            avg_volume = ts_mean(volume[1:], self.lookback)
            depth_proxy = avg_volume / volatility

            scores[symbol] = depth_proxy

        return scores
