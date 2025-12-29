"""
Momentum-based alpha factors.
"""

from typing import Dict

import numpy as np

from clyptq.data.store import DataView
from clyptq.factors.base import Factor
from clyptq.factors.ops import delay, delta, ts_mean, ts_std


class MomentumFactor(Factor):
    """Simple momentum: return over lookback period."""

    def __init__(self, lookback: int = 20):
        super().__init__("Momentum")
        self.lookback = lookback

    def compute(self, data: DataView) -> Dict[str, float]:
        return {
            s: delta(p, self.lookback - 1) / delay(p, self.lookback - 1)
            for s in data.symbols
            if len(p := data.close(s, self.lookback)) == self.lookback
        }


class RSIFactor(Factor):
    """RSI normalized to [-1, 1]."""

    def __init__(self, lookback: int = 14):
        super().__init__("RSI")
        self.lookback = lookback

    def compute(self, data: DataView) -> Dict[str, float]:
        scores = {}
        for s in data.symbols:
            try:
                p = data.close(s, self.lookback + 2)
                deltas = np.diff(p)
                gains = np.where(deltas > 0, deltas, 0)
                losses = np.where(deltas < 0, -deltas, 0)
                avg_gain = np.mean(gains)
                avg_loss = np.mean(losses)
                rsi = 100.0 if avg_loss < 1e-10 else 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
                scores[s] = (rsi - 50.0) / 50.0
            except (KeyError, ValueError):
                continue
        return scores


class TrendStrengthFactor(Factor):
    """Linear regression slope of log prices."""

    def __init__(self, lookback: int = 20):
        super().__init__("TrendStrength")
        self.lookback = lookback

    def compute(self, data: DataView) -> Dict[str, float]:
        return {
            s: np.polyfit(np.arange(len(p)), np.log(p), 1)[0] * 252
            for s in data.symbols
            if len(p := data.close(s, self.lookback)) == self.lookback
        }
