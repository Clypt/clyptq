"""
Momentum-based alpha factors.
"""

from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from clyptq.core.base import Factor, MultiTimeframeFactor
from clyptq.data.stores.store import DataView
from clyptq.trading.factors.ops import delay, delta


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


class MultiTimeframeMomentum(MultiTimeframeFactor):
    """Multi-timeframe momentum factor."""

    def __init__(
        self,
        timeframes: List[str] = ["1d", "1w"],
        lookbacks: Optional[Dict[str, int]] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__(timeframes, lookbacks)

        if weights is None:
            weight = 1.0 / len(timeframes)
            self.weights = {tf: weight for tf in timeframes}
        else:
            self.weights = weights

        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")

    def compute_mtf(
        self,
        symbol: str,
        timeframe_data: Dict[str, pd.DataFrame],
        timestamp: datetime,
    ) -> Optional[float]:
        """Compute weighted momentum across timeframes."""
        momentum_scores = {}

        for tf, data in timeframe_data.items():
            if len(data) < 2:
                return None

            first_price = data.iloc[0]["close"]
            last_price = data.iloc[-1]["close"]

            if first_price <= 0:
                return None

            momentum = (last_price - first_price) / first_price
            momentum_scores[tf] = momentum

        combined = sum(
            momentum_scores[tf] * self.weights[tf] for tf in self.timeframes
        )

        return combined
