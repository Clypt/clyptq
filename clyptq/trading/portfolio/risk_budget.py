"""
Risk Budgeting Portfolio Construction.

Portfolio construction based on risk contribution targets rather than
expected returns. More robust than mean-variance as it doesn't require
return forecasts.
"""

from typing import Dict, Optional, Literal
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from clyptq.core.base import PortfolioConstructor
from clyptq.core.types import Constraints


class RiskBudgetConstructor(PortfolioConstructor):
    """
    Risk Budget Portfolio Construction.

    Constructs portfolio weights by allocating risk according to specified
    budgets. Equal Risk Contribution (ERC) is a special case where all
    assets contribute equally to portfolio risk.

    Objective:
        Minimize: Σ(RC_i - target_i)²
        where RC_i = w_i * (Σw)_i / σ

    Usage:
        >>> rb = RiskBudgetConstructor(risk_model="sample_cov")
        >>> rb.fit(prices_df)
        >>> weights = rb.construct(scores, constraints)
    """

    def __init__(
        self,
        risk_model: Literal["sample_cov", "ledoit_wolf"] = "sample_cov",
        risk_budgets: Optional[Dict[str, float]] = None,
        lookback: int = 252,
    ):
        """
        Initialize Risk Budget constructor.

        Args:
            risk_model: Covariance estimation method
                - sample_cov: sample covariance matrix
                - ledoit_wolf: Ledoit-Wolf shrinkage
            risk_budgets: Target risk contribution for each asset
                If None, uses Equal Risk Contribution (ERC)
                Must sum to 1.0 if provided
            lookback: Lookback period for covariance estimation (trading days)
        """
        if lookback <= 0:
            raise ValueError(f"lookback must be positive, got {lookback}")

        if risk_budgets is not None:
            total = sum(risk_budgets.values())
            if not np.isclose(total, 1.0):
                raise ValueError(f"risk_budgets must sum to 1.0, got {total}")

        self.risk_model = risk_model
        self.risk_budgets = risk_budgets
        self.lookback = lookback

        self._cov_matrix: Optional[pd.DataFrame] = None
        self._fitted = False

    def fit(self, prices: pd.DataFrame) -> "RiskBudgetConstructor":
        """
        Estimate covariance matrix from price history.

        Args:
            prices: DataFrame with datetime index and symbol columns

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

        self._cov_matrix = self._estimate_covariance(returns_window)
        self._fitted = True

        return self

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
        Construct portfolio weights using risk budgeting.

        Args:
            scores: Factor scores {symbol: score} (used for filtering only)
            constraints: Portfolio constraints
            current_weights: Current portfolio weights (unused)

        Returns:
            Optimal portfolio weights {symbol: weight}
        """
        if not self._fitted:
            raise ValueError("Must call fit() before construct()")

        if not scores:
            return {}

        symbols = sorted(scores.keys())
        available_symbols = [s for s in symbols if s in self._cov_matrix.index]

        if not available_symbols:
            return {}

        n = len(available_symbols)
        Sigma = self._cov_matrix.loc[available_symbols, available_symbols].values

        if self.risk_budgets is None:
            target_budgets = np.ones(n) / n
        else:
            target_budgets = np.array([
                self.risk_budgets.get(s, 1.0 / n) for s in available_symbols
            ])
            target_budgets = target_budgets / target_budgets.sum()

        def risk_contributions(w):
            portfolio_var = w @ Sigma @ w
            if portfolio_var < 1e-10:
                return np.zeros(n)
            portfolio_vol = np.sqrt(portfolio_var)
            marginal_contrib = Sigma @ w / portfolio_vol
            return w * marginal_contrib

        def objective(w):
            rc = risk_contributions(w)
            portfolio_risk = np.sqrt(w @ Sigma @ w)
            if portfolio_risk < 1e-10:
                return 1e10
            rc_normalized = rc / portfolio_risk
            target_rc = target_budgets
            return np.sum((rc_normalized - target_rc) ** 2)

        bounds = [(0, constraints.max_position_size) for _ in range(n)]

        constraints_list = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        if constraints.max_gross_exposure < 1.0:
            constraints_list.append(
                {"type": "ineq", "fun": lambda w: constraints.max_gross_exposure - np.sum(w)}
            )

        eigenvalues = np.linalg.eigvals(Sigma)
        if np.any(eigenvalues < 1e-10):
            w0 = np.ones(n) / n
        else:
            inv_vol = 1.0 / np.sqrt(np.diag(Sigma))
            w0 = inv_vol / np.sum(inv_vol)

        result = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints_list,
            options={"maxiter": 2000, "ftol": 1e-9},
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

    def get_risk_contributions(
        self, weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate risk contribution for each asset.

        Args:
            weights: Portfolio weights {symbol: weight}

        Returns:
            Risk contributions {symbol: contribution}
        """
        if not self._fitted:
            raise ValueError("Must call fit() before get_risk_contributions()")

        symbols = sorted(weights.keys())
        w = np.array([weights[s] for s in symbols])
        Sigma = self._cov_matrix.loc[symbols, symbols].values

        portfolio_var = w @ Sigma @ w
        if portfolio_var < 1e-10:
            return {s: 0.0 for s in symbols}

        portfolio_vol = np.sqrt(portfolio_var)
        marginal_contrib = Sigma @ w / portfolio_vol
        risk_contrib = w * marginal_contrib

        total_rc = np.sum(risk_contrib)
        risk_contrib_pct = risk_contrib / total_rc if total_rc > 0 else risk_contrib

        return {s: float(risk_contrib_pct[i]) for i, s in enumerate(symbols)}
