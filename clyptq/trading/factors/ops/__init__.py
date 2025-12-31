"""Operations library for factor development."""

from clyptq.trading.factors.ops.cross_sectional import demean, normalize, rank, winsorize
from clyptq.trading.factors.ops.time_series import (
    correlation,
    delay,
    delta,
    ts_max,
    ts_mean,
    ts_min,
    ts_rank,
    ts_std,
    ts_sum,
)

__all__ = [
    "demean",
    "normalize",
    "rank",
    "winsorize",
    "correlation",
    "delay",
    "delta",
    "ts_max",
    "ts_mean",
    "ts_min",
    "ts_rank",
    "ts_std",
    "ts_sum",
]
