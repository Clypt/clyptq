import numpy as np
from typing import Dict

from clyptq.core.base import Factor
from clyptq.trading.factors.ops.time_series import ts_mean, ts_std


class RealizedSpreadFactor(Factor):
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


class PriceEfficiencyFactor(Factor):
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

            mid_price = (high + low) / 2.0
            deviation = np.abs(close - mid_price) / ((high - low) + 1e-10)
            avg_deviation = ts_mean(deviation, self.lookback)

            scores[symbol] = -avg_deviation

        return scores


class ImpliedBasisFactor(Factor):
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
            returns = np.diff(close) / close[:-1]

            if len(returns) < self.lookback:
                continue

            momentum = ts_mean(returns, self.lookback)
            volatility = ts_std(returns, self.lookback)

            if volatility < 1e-10:
                continue

            basis_proxy = momentum / volatility

            scores[symbol] = basis_proxy

        return scores
