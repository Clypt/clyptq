"""Library of pre-built alpha signals."""

from clyptq.strategy.signal.alpha.library.liquidity import (
    AmihudAlpha,
    EffectiveSpreadAlpha,
    VolatilityOfVolatilityAlpha,
)
from clyptq.strategy.signal.alpha.library.mean_reversion import (
    BollingerAlpha,
    PercentileAlpha,
    ZScoreAlpha,
)
from clyptq.strategy.signal.alpha.library.momentum import (
    MomentumAlpha,
    MultiTimeframeMomentumAlpha,
    RSIAlpha,
    TrendStrengthAlpha,
)
from clyptq.strategy.signal.alpha.library.quality import (
    MarketDepthProxyAlpha,
    PriceImpactAlpha,
    VolumeStabilityAlpha,
)
from clyptq.strategy.signal.alpha.library.size import DollarVolumeSizeAlpha
from clyptq.strategy.signal.alpha.library.value import (
    ImpliedBasisAlpha,
    PriceEfficiencyAlpha,
    RealizedSpreadAlpha,
)
from clyptq.strategy.signal.alpha.library.volatility import VolatilityAlpha
from clyptq.strategy.signal.alpha.library.volume import (
    DollarVolumeAlpha,
    VolumeAlpha,
    VolumeRatioAlpha,
)

__all__ = [
    # Liquidity
    "AmihudAlpha",
    "EffectiveSpreadAlpha",
    "VolatilityOfVolatilityAlpha",
    # Mean Reversion
    "BollingerAlpha",
    "PercentileAlpha",
    "ZScoreAlpha",
    # Momentum
    "MomentumAlpha",
    "MultiTimeframeMomentumAlpha",
    "RSIAlpha",
    "TrendStrengthAlpha",
    # Quality
    "MarketDepthProxyAlpha",
    "PriceImpactAlpha",
    "VolumeStabilityAlpha",
    # Size
    "DollarVolumeSizeAlpha",
    # Value
    "ImpliedBasisAlpha",
    "PriceEfficiencyAlpha",
    "RealizedSpreadAlpha",
    # Volatility
    "VolatilityAlpha",
    # Volume
    "DollarVolumeAlpha",
    "VolumeAlpha",
    "VolumeRatioAlpha",
]
