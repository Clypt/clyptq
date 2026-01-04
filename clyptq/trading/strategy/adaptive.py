"""
Adaptive Factor Weighting Strategy.

Dynamically adjusts factor weights based on recent performance,
allowing the strategy to adapt to changing market conditions.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from clyptq.core.base import Factor, PortfolioConstructor, Strategy
from clyptq.core.types import Constraints


class MarketRegime(Enum):
    HIGH_VOL = "high_volatility"
    LOW_VOL = "low_volatility"
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"


@dataclass
class RegimeConfig:
    volatility_threshold: float = 0.02
    trend_threshold: float = 0.01
    regime_lookback: int = 30


class _AdaptiveFactor(Factor):
    """Internal wrapper factor for AdaptiveStrategy."""

    def __init__(self, strategy: "AdaptiveStrategy"):
        self.strategy = strategy

    def compute(
        self,
        current_prices: pd.Series,
        history: pd.DataFrame,
        timestamp: datetime,
    ) -> Dict[str, float]:
        return self.strategy.compute_combined_scores(current_prices, history, timestamp)


class AdaptiveStrategy(Strategy):
    """
    Adaptive Factor Weighting Strategy.

    Adjusts factor weights dynamically based on recent performance metrics:
    - IC-weighted: Higher weight to factors with better predictive power
    - Sharpe-weighted: Higher weight to factors with better risk-adjusted returns
    - EMA-weighted: Exponential moving average of factor scores

    Usage:
        >>> factors = [MomentumFactor(), VolatilityFactor(), ValueFactor()]
        >>> strategy = AdaptiveStrategy(
        ...     factors=factors,
        ...     constructor=TopNConstructor(top_n=10),
        ...     weighting_method="ic_weighted",
        ...     lookback=60
        ... )
    """

    def __init__(
        self,
        factors_list: List[Factor],
        constructor: PortfolioConstructor,
        constraints_config: Constraints,
        weighting_method: Literal["ic_weighted", "sharpe_weighted", "ema_weighted"] = "ic_weighted",
        lookback: int = 60,
        min_weight: float = 0.05,
        ema_alpha: float = 0.1,
        rebalance_schedule: str = "weekly",
        warmup: int = 90,
    ):
        """
        Initialize Adaptive Strategy.

        Args:
            factors_list: List of factors to combine
            constructor: Portfolio constructor to use
            constraints_config: Portfolio constraints
            weighting_method: Method for calculating factor weights
                - ic_weighted: Information Coefficient based
                - sharpe_weighted: Sharpe ratio based
                - ema_weighted: Exponential moving average
            lookback: Lookback period for performance calculation (days)
            min_weight: Minimum weight for any factor (prevents zero weights)
            ema_alpha: Alpha parameter for EMA (0 < alpha < 1)
            rebalance_schedule: Rebalancing frequency
            warmup: Warmup periods required
        """
        if not factors_list:
            raise ValueError("factors_list cannot be empty")
        if lookback <= 0:
            raise ValueError(f"lookback must be positive, got {lookback}")
        if not 0 < min_weight < 1:
            raise ValueError(f"min_weight must be in (0, 1), got {min_weight}")
        if not 0 < ema_alpha < 1:
            raise ValueError(f"ema_alpha must be in (0, 1), got {ema_alpha}")

        self._factors = factors_list
        self._constructor = constructor
        self._constraints = constraints_config
        self.weighting_method = weighting_method
        self.lookback = lookback
        self.min_weight = min_weight
        self.ema_alpha = ema_alpha
        self._schedule = rebalance_schedule
        self._warmup = warmup

        self._factor_weights: Dict[str, float] = {
            f"factor_{i}": 1.0 / len(factors_list) for i in range(len(factors_list))
        }
        self._factor_history: Dict[str, List[Dict[str, float]]] = {
            f"factor_{i}": [] for i in range(len(factors_list))
        }
        self._price_history: List[pd.Series] = []
        self._timestamp_history: List[datetime] = []

    @property
    def name(self) -> str:
        return f"Adaptive-{self.weighting_method}-{len(self._factors)}F"

    def factors(self) -> List[Factor]:
        return self._factors

    def portfolio_constructor(self) -> PortfolioConstructor:
        return self._constructor

    def constraints(self) -> Constraints:
        return self._constraints

    def schedule(self) -> str:
        return self._schedule

    def warmup_periods(self) -> int:
        return self._warmup

    def compute_combined_scores(
        self,
        current_prices: pd.Series,
        history: pd.DataFrame,
        timestamp: datetime,
    ) -> Dict[str, float]:
        """
        Compute combined factor scores with adaptive weights.

        Args:
            current_prices: Current prices for all symbols
            history: Historical price data
            timestamp: Current timestamp

        Returns:
            Combined scores {symbol: score}
        """
        factor_scores_list = []
        for i, factor in enumerate(self._factors):
            scores = factor.compute(current_prices, history, timestamp)
            factor_scores_list.append(scores)
            self._factor_history[f"factor_{i}"].append(scores)

        self._price_history.append(current_prices)
        self._timestamp_history.append(timestamp)

        if len(self._price_history) >= self.lookback:
            self._update_factor_weights()

        combined_scores = self._combine_scores(factor_scores_list)
        return combined_scores

    def _update_factor_weights(self):
        """Update factor weights based on recent performance."""
        if self.weighting_method == "ic_weighted":
            self._factor_weights = self._calculate_ic_weights()
        elif self.weighting_method == "sharpe_weighted":
            self._factor_weights = self._calculate_sharpe_weights()
        elif self.weighting_method == "ema_weighted":
            self._factor_weights = self._calculate_ema_weights()

    def _calculate_ic_weights(self) -> Dict[str, float]:
        """Calculate weights based on Information Coefficient."""
        lookback_start = max(0, len(self._price_history) - self.lookback)
        recent_prices = self._price_history[lookback_start:]
        recent_timestamps = self._timestamp_history[lookback_start:]

        if len(recent_prices) < 2:
            return {f"factor_{i}": 1.0 / len(self._factors) for i in range(len(self._factors))}

        returns = []
        for i in range(1, len(recent_prices)):
            ret = recent_prices[i] / recent_prices[i - 1] - 1
            returns.append(ret)

        ic_scores = []
        for i in range(len(self._factors)):
            factor_key = f"factor_{i}"
            recent_factor_scores = self._factor_history[factor_key][lookback_start:]

            if len(recent_factor_scores) < 2:
                ic_scores.append(0.0)
                continue

            correlations = []
            for j in range(len(returns)):
                if j >= len(recent_factor_scores) - 1:
                    break

                scores = recent_factor_scores[j]
                future_returns = returns[j]

                common_symbols = set(scores.keys()) & set(future_returns.index)
                if len(common_symbols) < 3:
                    continue

                score_values = [scores[s] for s in common_symbols]
                return_values = [future_returns[s] for s in common_symbols]

                if len(score_values) > 0:
                    corr = np.corrcoef(score_values, return_values)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)

            ic = np.mean(correlations) if correlations else 0.0
            ic_scores.append(abs(ic))

        ic_scores = np.array(ic_scores)
        if np.sum(ic_scores) < 1e-10:
            weights = np.ones(len(self._factors)) / len(self._factors)
        else:
            weights = ic_scores / np.sum(ic_scores)
            weights = np.maximum(weights, self.min_weight)
            weights = weights / np.sum(weights)

        return {f"factor_{i}": float(weights[i]) for i in range(len(self._factors))}

    def _calculate_sharpe_weights(self) -> Dict[str, float]:
        """Calculate weights based on Sharpe ratio of factor returns."""
        lookback_start = max(0, len(self._price_history) - self.lookback)

        if len(self._price_history) < 2:
            return {f"factor_{i}": 1.0 / len(self._factors) for i in range(len(self._factors))}

        sharpe_scores = []
        for i in range(len(self._factors)):
            factor_key = f"factor_{i}"
            recent_factor_scores = self._factor_history[factor_key][lookback_start:]

            if len(recent_factor_scores) < 2:
                sharpe_scores.append(0.0)
                continue

            factor_returns = []
            for j in range(1, len(recent_factor_scores)):
                prev_scores = recent_factor_scores[j - 1]
                curr_scores = recent_factor_scores[j]

                common_symbols = set(prev_scores.keys()) & set(curr_scores.keys())
                if not common_symbols:
                    continue

                prev_mean = np.mean([prev_scores[s] for s in common_symbols])
                curr_mean = np.mean([curr_scores[s] for s in common_symbols])

                if abs(prev_mean) > 1e-10:
                    factor_return = (curr_mean - prev_mean) / abs(prev_mean)
                    factor_returns.append(factor_return)

            if len(factor_returns) > 0:
                mean_return = np.mean(factor_returns)
                std_return = np.std(factor_returns)
                sharpe = mean_return / std_return if std_return > 1e-10 else 0.0
                sharpe_scores.append(max(sharpe, 0.0))
            else:
                sharpe_scores.append(0.0)

        sharpe_scores = np.array(sharpe_scores)
        if np.sum(sharpe_scores) < 1e-10:
            weights = np.ones(len(self._factors)) / len(self._factors)
        else:
            weights = sharpe_scores / np.sum(sharpe_scores)
            weights = np.maximum(weights, self.min_weight)
            weights = weights / np.sum(weights)

        return {f"factor_{i}": float(weights[i]) for i in range(len(self._factors))}

    def _calculate_ema_weights(self) -> Dict[str, float]:
        """Calculate weights using Exponential Moving Average."""
        if len(self._factor_history[f"factor_0"]) < 2:
            return {f"factor_{i}": 1.0 / len(self._factors) for i in range(len(self._factors))}

        ema_magnitudes = []
        for i in range(len(self._factors)):
            factor_key = f"factor_{i}"
            recent_scores = self._factor_history[factor_key][-self.lookback :]

            magnitudes = []
            for scores in recent_scores:
                if scores:
                    mag = np.mean([abs(v) for v in scores.values()])
                    magnitudes.append(mag)

            if magnitudes:
                ema = magnitudes[0]
                for mag in magnitudes[1:]:
                    ema = self.ema_alpha * mag + (1 - self.ema_alpha) * ema
                ema_magnitudes.append(ema)
            else:
                ema_magnitudes.append(0.0)

        ema_magnitudes = np.array(ema_magnitudes)
        if np.sum(ema_magnitudes) < 1e-10:
            weights = np.ones(len(self._factors)) / len(self._factors)
        else:
            weights = ema_magnitudes / np.sum(ema_magnitudes)
            weights = np.maximum(weights, self.min_weight)
            weights = weights / np.sum(weights)

        return {f"factor_{i}": float(weights[i]) for i in range(len(self._factors))}

    def _combine_scores(self, factor_scores_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Combine individual factor scores using current weights."""
        all_symbols = set()
        for scores in factor_scores_list:
            all_symbols.update(scores.keys())

        combined = {}
        for symbol in all_symbols:
            weighted_sum = 0.0
            total_weight = 0.0

            for i, scores in enumerate(factor_scores_list):
                if symbol in scores:
                    weight = self._factor_weights[f"factor_{i}"]
                    weighted_sum += weight * scores[symbol]
                    total_weight += weight

            if total_weight > 0:
                combined[symbol] = weighted_sum / total_weight
            else:
                combined[symbol] = 0.0

        return combined

    def get_factor_weights(self) -> Dict[str, float]:
        """Get current factor weights."""
        return self._factor_weights.copy()
