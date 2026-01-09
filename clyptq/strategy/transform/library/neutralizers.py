"""Neutralization transforms."""

from clyptq.strategy.transform.base import BaseTransform
from clyptq import operator
from clyptq.operator.linalg import linalg_lstsq, matmul


class Demean(BaseTransform):
    """Remove cross-sectional mean."""

    def compute(self, data):
        return operator.demean(data)


class SectorNeutralizer(BaseTransform):
    """Neutralize within each sector."""

    def __init__(self, sector_mapping):
        self.sector_mapping = sector_mapping

    def compute(self, data):
        sectors = operator.align_to_index(self.sector_mapping, data, default="Unknown")
        return operator.sector_neutralize(data, sectors)


class FactorNeutralizer(BaseTransform):
    """Remove factor exposures via OLS regression."""

    def __init__(self, factor_matrix):
        self.factor_matrix = factor_matrix

    def compute(self, data):
        B_aligned = operator.reindex(self.factor_matrix, data)
        coeffs = linalg_lstsq(B_aligned, data)
        projection = matmul(B_aligned, coeffs)
        return operator.sub(data, projection)


class BetaNeutralizer(BaseTransform):
    """Neutralize market beta exposure."""

    def __init__(self, betas):
        self.betas = betas

    def compute(self, data):
        betas = operator.align_to_index(self.betas, data, default=1.0)

        # Formula: neutral = data - (sum(data * beta) / sum(beta)) * beta
        # This removes the beta-weighted average from each row

        data_times_beta = operator.mul(data, betas)
        weighted_sum = operator.cs_sum(data_times_beta)  # Series (T,)
        total_beta = operator.cs_sum(betas)  # scalar

        adjustment = operator.div(weighted_sum, total_beta)  # Series (T,)

        # Multiply adjustment(T,) by betas(N,) to get DataFrame (T x N)
        beta_component = operator.outer_mul(adjustment, betas)

        return operator.sub(data, beta_component)
