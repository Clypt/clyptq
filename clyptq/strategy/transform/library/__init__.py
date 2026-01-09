"""Transform library - pre-built transforms for common use cases.

All transforms: DataFrame/Series → DataFrame/Series (operator-compatible)

Neutralizers:
- Demean: Remove cross-sectional mean
- FactorNeutralizer: Remove factor exposures via OLS
- SectorNeutralizer: Neutralize within sectors
- BetaNeutralizer: Remove market beta exposure

Normalizers (sum to 1):
- L1Norm: Normalize to sum to 1 (portfolio weights)
- L2Norm: Normalize to unit L2 norm
- Softmax: Exp-weighted sum to 1

Scalers:
- RankNormalizer: Rank normalize to [0, 1]
- ZScoreNormalizer: Z-score normalize (mean=0, std=1)
- Winsorizer: Clip outliers to percentile bounds
- VolatilityScaler: Scale by inverse volatility

Optimizers:
- MeanVarianceOptimizer: Mean-Variance Optimization
- RiskParityOptimizer: Risk Parity weights
- ClipWeights: Clip weights to bounds
- MaxPositions: Limit number of positions
"""

# Neutralizers
from clyptq.strategy.transform.library.neutralizers import (
    Demean,
    FactorNeutralizer,
    SectorNeutralizer,
    BetaNeutralizer,
)

# Scalers / Normalizers
from clyptq.strategy.transform.library.scalers import (
    L1Norm,
    L2Norm,
    Softmax,
    RankNormalizer,
    ZScoreNormalizer,
    Winsorizer,
    VolatilityScaler,
)

# Optimizers / Constraints
from clyptq.strategy.transform.library.optimizers import (
    MeanVarianceOptimizer,
    RiskParityOptimizer,
    ClipWeights,
    MaxPositions,
)

__all__ = [
    # Neutralizers
    "Demean",
    "FactorNeutralizer",
    "SectorNeutralizer",
    "BetaNeutralizer",
    # Normalizers
    "L1Norm",
    "L2Norm",
    "Softmax",
    # Scalers
    "RankNormalizer",
    "ZScoreNormalizer",
    "Winsorizer",
    "VolatilityScaler",
    # Optimizers / Constraints
    "MeanVarianceOptimizer",
    "RiskParityOptimizer",
    "ClipWeights",
    "MaxPositions",
]
