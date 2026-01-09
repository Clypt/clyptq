"""Portfolio optimization transforms."""

from clyptq.strategy.transform.base import BaseTransform
from clyptq import operator
from clyptq.operator.linalg import linalg_inv, matmul


class MeanVarianceOptimizer(BaseTransform):
    """Mean-Variance Optimization (Markowitz)."""

    def __init__(self, expected_returns, cov_matrix, risk_aversion=1.0, regularize=1e-6):
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.risk_aversion = risk_aversion
        self.regularize = regularize

    def compute(self, data):
        mu_aligned = operator.reindex(self.expected_returns, data, fill_value=0)
        cov_aligned = operator.reindex_2d(self.cov_matrix, data, fill_value=0)

        cov_inv = linalg_inv(cov_aligned, regularize=self.regularize)
        raw_weights = operator.div(matmul(cov_inv, mu_aligned), self.risk_aversion)

        return operator.l1_norm(raw_weights)


class RiskParityOptimizer(BaseTransform):
    """Risk Parity weights."""

    def __init__(self, volatilities, min_vol=0.01):
        self.volatilities = volatilities
        self.min_vol = min_vol

    def compute(self, data):
        vols = operator.align_to_index(self.volatilities, data, default=0.2)
        vols_clipped = operator.clip(vols, lower=self.min_vol)
        inv_vols = operator.div(1.0, vols_clipped)
        return operator.l1_norm(inv_vols)


class ClipWeights(BaseTransform):
    """Clip weights to bounds."""

    def __init__(self, lower=-1.0, upper=1.0, renormalize=True):
        self.lower = lower
        self.upper = upper
        self.renormalize = renormalize

    def compute(self, data):
        clipped = operator.clip(data, lower=self.lower, upper=self.upper)
        if self.renormalize:
            return operator.l1_norm(clipped)
        return clipped


class MaxPositions(BaseTransform):
    """Limit number of positions."""

    def __init__(self, n=20, renormalize=True):
        self.n = n
        self.renormalize = renormalize

    def compute(self, data):
        mask = operator.topk_mask(operator.abs(data), k=self.n)
        sparse = operator.mul(data, mask)
        if self.renormalize:
            return operator.l1_norm(sparse)
        return sparse
