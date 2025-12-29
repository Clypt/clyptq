"""
Mean reversion alpha factors.
"""

from typing import Dict

import numpy as np

from clyptq.data.store import DataView
from clyptq.factors.base import Factor
from clyptq.factors.ops import ts_mean, ts_rank, ts_std


class BollingerFactor(Factor):
    """Bollinger mean reversion: (middle - price) / bandwidth."""

    def __init__(self, lookback: int = 20, num_std: float = 2.0):
        super().__init__("Bollinger")
        self.lookback = lookback
        self.num_std = num_std

    def compute(self, data: DataView) -> Dict[str, float]:
        scores = {}
        for s in data.symbols:
            try:
                p = data.close(s, self.lookback)
                mid = ts_mean(p, self.lookback)
                std = ts_std(p, self.lookback)
                bw = self.num_std * std
                scores[s] = 0.0 if bw < 1e-10 else np.clip((mid - p[-1]) / bw, -1.0, 1.0)
            except (KeyError, ValueError):
                continue
        return scores


class ZScoreFactor(Factor):
    """Z-Score mean reversion: -(price - mean) / std."""

    def __init__(self, lookback: int = 20):
        super().__init__("ZScore")
        self.lookback = lookback

    def compute(self, data: DataView) -> Dict[str, float]:
        scores = {}
        for s in data.symbols:
            try:
                p = data.close(s, self.lookback)
                mean = ts_mean(p, self.lookback)
                std = ts_std(p, self.lookback)
                scores[s] = 0.0 if std < 1e-10 else np.clip(-(p[-1] - mean) / std, -3.0, 3.0)
            except (KeyError, ValueError):
                continue
        return scores


class PercentileFactor(Factor):
    """Percentile mean reversion: -(percentile - 0.5) * 2."""

    def __init__(self, lookback: int = 20):
        super().__init__("Percentile")
        self.lookback = lookback

    def compute(self, data: DataView) -> Dict[str, float]:
        return {
            s: -(ts_rank(p, self.lookback) - 0.5) * 2
            for s in data.symbols
            if len(p := data.close(s, self.lookback)) == self.lookback
        }
