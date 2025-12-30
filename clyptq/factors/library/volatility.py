"""
Volatility-based alpha factors.
"""

from typing import Dict

import numpy as np

from clyptq.data.stores.store import DataView
from clyptq.core.base import Factor
from clyptq.factors.ops import rank, ts_std


class VolatilityFactor(Factor):
    """Inverse volatility: prefer low volatility assets."""

    def __init__(self, lookback: int = 20):
        super().__init__("Volatility")
        self.lookback = lookback

    def compute(self, data: DataView) -> Dict[str, float]:
        scores = {
            s: -ts_std(data.returns(s, self.lookback), self.lookback)
            for s in data.symbols
            if len(data.close(s, self.lookback)) == self.lookback
        }
        return rank(scores)
