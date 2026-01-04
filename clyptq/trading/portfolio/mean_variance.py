"""
Mean-Variance Optimization (Markowitz).

Portfolio optimization based on expected returns and risk (covariance).
"""

from typing import Dict, Optional, Literal
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from clyptq.core.base import PortfolioConstructor
from clyptq.core.types import Constraints


class MeanVarianceConstructor(PortfolioConstructor):
    """
    Mean-Variance Portfolio Optimization (Markowitz).

    Constructs optimal portfolio weights by maximizing expected return
    while minimizing risk (variance), subject to constraints.

    Objective:
        min: λ * w^T Σ w - w^T μ + γ * ||w - w_prev||²
        s.t: Σw = 1, 0 ≤ w ≤ max_position_size

    where:
        - w: portfolio weights
        - Σ: covariance matrix
        - μ: expected returns
        - λ: risk aversion coefficient
        - γ: turnover penalty coefficient
        - w_prev: current portfolio weights

    Usage:
        >>> mv = MeanVarianceConstructor(risk_aversion=2.0, lookback=252)
        >>> mv.fit(prices_df)  # Estimate returns and covariance
        >>> weights = mv.construct(scores, constraints, current_weights)
    """

    def __init__(
        self,
        return_model: Literal["historical", "factor_model", "shrinkage"] = "historical",
        risk_model: Literal["sample_cov", "ledoit_wolf"] = "sample_cov",
        target_return: Optional[float] = None,
        risk_aversion: float = 1.0,
        turnover_penalty: float = 0.0,
        lookback: int = 252,
    ):
        """
        Initialize Mean-Variance constructor.

        Args:
            return_model: Return estimation method
                - historical: sample mean of returns
                - factor_model: returns from factor scores
                - shrinkage: James-Stein shrinkage estimator
            risk_model: Covariance estimation method
                - sample_cov: sample covariance matrix
                - ledoit_wolf: Ledoit-Wolf shrinkage
            target_return: Minimum target return (if None, unconstrained)
            risk_aversion: Risk aversion coefficient (λ ≥ 0)
                - λ=0: maximum return (ignores risk)
                - λ>0: trade-off between return and risk
                - λ→∞: minimum variance portfolio
            turnover_penalty: Turnover penalty coefficient (γ ≥ 0)
                - γ=0: no turnover penalty
                - γ>0: penalize deviation from current weights
            lookback: Lookback period for estimation (trading days)
        """
        if risk_aversion < 0:
            raise ValueError(f"risk_aversion must be >= 0, got {risk_aversion}")
        if turnover_penalty < 0:
            raise ValueError(f"turnover_penalty must be >= 0, got {turnover_penalty}")
        if lookback <= 0:
            raise ValueError(f"lookback must be positive, got {lookback}")

        self.return_model = return_model
        self.risk_model = risk_model
        self.target_return = target_return
        self.risk_aversion = risk_aversion
        self.turnover_penalty = turnover_penalty
        self.lookback = lookback

        self._expected_returns: Optional[pd.Series] = None
        self._cov_matrix: Optional[pd.DataFrame] = None
        self._fitted = False

    def fit(self, prices: pd.DataFrame) -> "MeanVarianceConstructor":
        """
        Estimate expected returns and covariance from price history.

        Args:
            prices: DataFrame with datetime index and symbol columns
                Example:
                            BTC/USDT  ETH/USDT  SOL/USDT
                2024-01-01     50000     3000       100
                2024-01-02     51000     3100       102

        Returns:
            self (for method chaining)
        """
        if prices.empty:
            raise ValueError("prices DataFrame is empty")

        if len(prices) < self.lookback:
            warnings.warn(
                f"Price history ({len(prices)} days) is shorter than lookback "
                f"({self.lookback} days). Using all available data.",
                UserWarning,
            )

        returns = prices.pct_change().dropna()
        returns_window = returns.iloc[-self.lookback :]

        self._expected_returns = self._estimate_returns(returns_window)
        self._cov_matrix = self._estimate_covariance(returns_window)
        self._fitted = True

        return self

    def _estimate_returns(self, returns: pd.DataFrame) -> pd.Series:
        """Estimate expected returns based on return_model."""
        if self.return_model == "historical":
            return returns.mean()

        elif self.return_model == "shrinkage":
            sample_mean = returns.mean()
            grand_mean = sample_mean.mean()
            return 0.5 * sample_mean + 0.5 * grand_mean

        elif self.return_model == "factor_model":
            return returns.mean()

        else:
            raise ValueError(f"Unknown return_model: {self.return_model}")

    def _estimate_covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Estimate covariance matrix based on risk_model."""
        if self.risk_model == "sample_cov":
            return returns.cov()

        elif self.risk_model == "ledoit_wolf":
            from sklearn.covariance import LedoitWolf

            lw = LedoitWolf()
            lw.fit(returns)
            return pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)

        else:
            raise ValueError(f"Unknown risk_model: {self.risk_model}")

    def construct(
        self,
        scores: Dict[str, float],
        constraints: Constraints,
        current_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Construct optimal portfolio weights.

        Args:
            scores: Factor scores {symbol: score} (used for factor_model)
            constraints: Portfolio constraints
            current_weights: Current portfolio weights (for turnover penalty)

        Returns:
            Optimal portfolio weights {symbol: weight}
        """
        if not self._fitted:
            raise ValueError("Must call fit() before construct()")

        if not scores:
            return {}

        symbols = sorted(scores.keys())
        available_symbols = [s for s in symbols if s in self._expected_returns.index]

        if not available_symbols:
            return {}

        n = len(available_symbols)

        mu = self._expected_returns[available_symbols].values
        Sigma = self._cov_matrix.loc[available_symbols, available_symbols].values

        if self.return_model == "factor_model":
            factor_scores = np.array([scores[s] for s in available_symbols])
            mu = factor_scores / (np.abs(factor_scores).sum() + 1e-10)

        w_prev = np.zeros(n)
        if current_weights and self.turnover_penalty > 0:
            for i, symbol in enumerate(available_symbols):
                w_prev[i] = current_weights.get(symbol, 0.0)

        def objective(w):
            portfolio_variance = w @ Sigma @ w
            portfolio_return = w @ mu
            turnover = np.sum(np.abs(w - w_prev))

            return (
                self.risk_aversion * portfolio_variance
                - portfolio_return
                + self.turnover_penalty * turnover
            )

        bounds = [(0, constraints.max_position_size) for _ in range(n)]

        constraints_list = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        if self.target_return is not None:
            constraints_list.append(
                {"type": "ineq", "fun": lambda w: w @ mu - self.target_return}
            )

        if constraints.max_gross_exposure < 1.0:
            constraints_list.append(
                {"type": "ineq", "fun": lambda w: constraints.max_gross_exposure - np.sum(w)}
            )

        w0 = np.ones(n) / n

        result = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints_list,
            options={"maxiter": 1000},
        )

        if not result.success:
            warnings.warn(f"Optimization failed: {result.message}", UserWarning)
            return {}

        optimal_weights = result.x

        weights_dict = {}
        for i, symbol in enumerate(available_symbols):
            if optimal_weights[i] > constraints.min_position_size:
                weights_dict[symbol] = float(optimal_weights[i])

        return weights_dict
