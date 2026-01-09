"""Alpha signals (role=ALPHA).

Alpha is a Signal for return generation.
Inherits from BaseSignal with role=SignalRole.ALPHA.

Usage:
    from clyptq.strategy.signal.alpha import MomentumAlpha, RSIAlpha
    from clyptq.strategy.signal import Signal

    signal = Signal(MomentumAlpha(lookback=20)).pipe(Demean())
    scores = signal.compute(data).value
"""

# Base classes from parent
from clyptq.strategy.signal.base import (
    BaseSignal,
    Signal,
    SignalRole,
)

# Legacy aliases
BaseAlpha = BaseSignal
Alpha = Signal

# Pre-built alphas from library
from clyptq.strategy.signal.alpha.library import (
    # Liquidity
    AmihudAlpha,
    EffectiveSpreadAlpha,
    VolatilityOfVolatilityAlpha,
    # Mean Reversion
    BollingerAlpha,
    PercentileAlpha,
    ZScoreAlpha,
    # Momentum
    MomentumAlpha,
    MultiTimeframeMomentumAlpha,
    RSIAlpha,
    TrendStrengthAlpha,
    # Quality
    MarketDepthProxyAlpha,
    PriceImpactAlpha,
    VolumeStabilityAlpha,
    # Size
    DollarVolumeSizeAlpha,
    # Value
    ImpliedBasisAlpha,
    PriceEfficiencyAlpha,
    RealizedSpreadAlpha,
    # Volatility
    VolatilityAlpha,
    # Volume
    DollarVolumeAlpha,
    VolumeAlpha,
    VolumeRatioAlpha,
)

__all__ = [
    # Base
    "BaseSignal",
    "Signal",
    "SignalRole",
    # Legacy aliases
    "BaseAlpha",
    "Alpha",
    # Pre-built alphas
    "AmihudAlpha",
    "EffectiveSpreadAlpha",
    "VolatilityOfVolatilityAlpha",
    "BollingerAlpha",
    "PercentileAlpha",
    "ZScoreAlpha",
    "MomentumAlpha",
    "MultiTimeframeMomentumAlpha",
    "RSIAlpha",
    "TrendStrengthAlpha",
    "MarketDepthProxyAlpha",
    "PriceImpactAlpha",
    "VolumeStabilityAlpha",
    "DollarVolumeSizeAlpha",
    "ImpliedBasisAlpha",
    "PriceEfficiencyAlpha",
    "RealizedSpreadAlpha",
    "VolatilityAlpha",
    "DollarVolumeAlpha",
    "VolumeAlpha",
    "VolumeRatioAlpha",
]
