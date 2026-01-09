"""Scaling transforms."""

from clyptq.strategy.transform.base import BaseTransform
from clyptq import operator


class L1Norm(BaseTransform):
    """L1 normalization - scale to sum to 1."""

    def __init__(self, long_only=False):
        self.long_only = long_only

    def compute(self, data):
        if self.long_only:
            data = operator.clip(data, lower=0)
        return operator.l1_norm(data)


class L2Norm(BaseTransform):
    """L2 normalization - scale to unit norm."""

    def compute(self, data):
        return operator.l2_norm(data)


class Softmax(BaseTransform):
    """Softmax normalization."""

    def __init__(self, temperature=1.0):
        self.temperature = temperature

    def compute(self, data):
        return operator.softmax(data, temperature=self.temperature)


class RankNormalizer(BaseTransform):
    """Rank normalization to [0, 1]."""

    def compute(self, data):
        # operator.rank already returns percentile ranks (0 to 1)
        return operator.rank(data)


class ZScoreNormalizer(BaseTransform):
    """Z-Score normalization."""

    def compute(self, data):
        return operator.zscore(data)


class Winsorizer(BaseTransform):
    """Winsorize outliers using standard deviation bounds.

    Caps values at mean +/- std_mult * std.
    """

    def __init__(self, std_mult: float = 3.0):
        self.std_mult = std_mult

    def compute(self, data):
        return operator.winsorize(data, std_mult=self.std_mult)


class VolatilityScaler(BaseTransform):
    """Scale by inverse volatility."""

    def __init__(self, volatilities, target_vol=0.20, min_vol=0.01):
        self.volatilities = volatilities
        self.target_vol = target_vol
        self.min_vol = min_vol

    def compute(self, data):
        vols = operator.align_to_index(self.volatilities, data, default=self.target_vol)
        vols_clipped = operator.clip(vols, lower=self.min_vol)
        scale_factors = operator.div(self.target_vol, vols_clipped)
        return operator.mul(data, scale_factors)
