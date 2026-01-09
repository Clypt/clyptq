"""Factor signals (role=FACTOR).

Factor is a Signal for risk explanation.
Inherits from BaseSignal with role=SignalRole.FACTOR.
normalize=True is the default (cross-sectional z-score).

Usage:
    from clyptq.strategy.signal.factor import MomentumFactor, VolatilityFactor
    from clyptq import operator

    # Factor calculation
    momentum = MomentumFactor(lookback=20).compute(data)

    # Neutralization
    alpha_neutral = operator.neutralize(alpha, [momentum])
"""

# Base classes from parent
from clyptq.strategy.signal.base import (
    BaseSignal,
    Signal,
    SignalRole,
)

# Legacy aliases
BaseFactor = BaseSignal

# Re-export from library
from clyptq.strategy.signal.factor.library import (
    MomentumFactor,
    ShortTermReversalFactor,
    VolatilityFactor,
    SizeFactor,
    MeanReversionFactor,
    VolumeTrendFactor,
    LiquidityFactor,
    QualityFactor,
)

__all__ = [
    # Base
    "BaseSignal",
    "Signal",
    "SignalRole",
    # Legacy aliases
    "BaseFactor",
    # Library
    "MomentumFactor",
    "ShortTermReversalFactor",
    "VolatilityFactor",
    "SizeFactor",
    "MeanReversionFactor",
    "VolumeTrendFactor",
    "LiquidityFactor",
    "QualityFactor",
]
