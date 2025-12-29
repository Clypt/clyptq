"""
Rolling performance metrics.

Calculate time-series metrics over rolling windows to analyze
how strategy performance changes over time.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd

from clyptq.core.types import BacktestResult


@dataclass
class RollingMetricsResult:
    """Rolling metrics result."""

    timestamps: List[datetime]
    sharpe_ratio: List[float]
    sortino_ratio: List[float]
    volatility: List[float]
    max_drawdown: List[float]
    returns: List[float]

    def to_dict(self) -> dict:
        return {
            "timestamps": [ts.isoformat() for ts in self.timestamps],
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "volatility": self.volatility,
            "max_drawdown": self.max_drawdown,
            "returns": self.returns,
        }


class RollingMetricsCalculator:
    """
    Calculate rolling performance metrics.

    Provides rolling window calculations of:
    - Sharpe ratio
    - Sortino ratio
    - Volatility
    - Max drawdown
    """

    def __init__(self, window: int = 30, risk_free_rate: float = 0.0):
        """
        Initialize calculator.

        Args:
            window: Rolling window size in days
            risk_free_rate: Annualized risk-free rate
        """
        self.window = window
        self.risk_free_rate = risk_free_rate

    def calculate(self, result: BacktestResult) -> RollingMetricsResult:
        """
        Calculate rolling metrics.

        Args:
            result: Backtest result with snapshots

        Returns:
            RollingMetricsResult with time-series metrics
        """
        if not result.snapshots or len(result.snapshots) < self.window:
            raise ValueError(f"Need at least {self.window} snapshots")

        equity = np.array([s.equity for s in result.snapshots])
        timestamps = [s.timestamp for s in result.snapshots]

        returns = np.diff(equity) / equity[:-1]

        rolling_sharpe = self._rolling_sharpe(returns)
        rolling_sortino = self._rolling_sortino(returns)
        rolling_vol = self._rolling_volatility(returns)
        rolling_dd = self._rolling_max_drawdown(equity)

        valid_idx = self.window
        valid_timestamps = timestamps[valid_idx:]
        valid_returns = returns[valid_idx - 1 :].tolist()

        return RollingMetricsResult(
            timestamps=valid_timestamps,
            sharpe_ratio=rolling_sharpe,
            sortino_ratio=rolling_sortino,
            volatility=rolling_vol,
            max_drawdown=rolling_dd,
            returns=valid_returns,
        )

    def _rolling_sharpe(self, returns: np.ndarray) -> List[float]:
        """Calculate rolling Sharpe ratio."""
        sharpe = []
        for i in range(self.window - 1, len(returns)):
            window_returns = returns[i - self.window + 1 : i + 1]
            mean_return = np.mean(window_returns)
            std_return = np.std(window_returns, ddof=1)

            if std_return == 0:
                sharpe.append(0.0)
            else:
                annual_sharpe = (
                    (mean_return - self.risk_free_rate / 365) / std_return * np.sqrt(365)
                )
                sharpe.append(annual_sharpe)

        return sharpe

    def _rolling_sortino(self, returns: np.ndarray) -> List[float]:
        """Calculate rolling Sortino ratio."""
        sortino = []
        for i in range(self.window - 1, len(returns)):
            window_returns = returns[i - self.window + 1 : i + 1]
            mean_return = np.mean(window_returns)

            downside_returns = window_returns[window_returns < 0]
            if len(downside_returns) == 0:
                sortino.append(0.0)
                continue

            downside_std = np.std(downside_returns, ddof=1)
            if downside_std == 0:
                sortino.append(0.0)
            else:
                annual_sortino = (
                    (mean_return - self.risk_free_rate / 365) / downside_std * np.sqrt(365)
                )
                sortino.append(annual_sortino)

        return sortino

    def _rolling_volatility(self, returns: np.ndarray) -> List[float]:
        """Calculate rolling volatility."""
        volatility = []
        for i in range(self.window - 1, len(returns)):
            window_returns = returns[i - self.window + 1 : i + 1]
            std_return = np.std(window_returns, ddof=1)
            annual_vol = std_return * np.sqrt(365)
            volatility.append(annual_vol)

        return volatility

    def _rolling_max_drawdown(self, equity: np.ndarray) -> List[float]:
        """Calculate rolling max drawdown."""
        max_dd = []
        for i in range(self.window, len(equity)):
            window_equity = equity[i - self.window : i]
            running_max = np.maximum.accumulate(window_equity)
            drawdown = (window_equity - running_max) / running_max
            max_dd.append(float(np.min(drawdown)))

        return max_dd
