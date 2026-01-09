"""
Summary statistics for final analysis.

Simple statistical summary functions. For complex operations, use clyptq.operator.

Usage:
    from clyptq.analytics import ic, sharpe, max_drawdown

    # Signal quality
    ic_val = ic(signal, returns)

    # Return metrics
    sr = sharpe(portfolio_returns)
    mdd = max_drawdown(portfolio_returns)
"""

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd


def ic(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    method: str = "spearman",
) -> pd.Series:
    """
    Compute cross-sectional IC (Information Coefficient).

    Args:
        signal: Signal scores (T x N)
        returns: Forward returns (T x N)
        method: "spearman" or "pearson"

    Returns:
        IC time series
    """
    ic_values = []

    for idx in signal.index:
        if idx not in returns.index:
            ic_values.append(np.nan)
            continue

        sig = signal.loc[idx].dropna()
        ret = returns.loc[idx].dropna()
        common = sig.index.intersection(ret.index)

        if len(common) < 5:
            ic_values.append(np.nan)
            continue

        sig, ret = sig[common], ret[common]

        if method == "spearman":
            corr = sig.rank().corr(ret.rank())
        else:
            corr = sig.corr(ret)

        ic_values.append(corr)

    return pd.Series(ic_values, index=signal.index, name="IC")


@dataclass
class ICStats:
    """IC summary statistics."""
    mean: float
    std: float
    ir: float  # IC / std(IC)
    hit_rate: float  # % positive
    t_stat: float

    def __repr__(self) -> str:
        return (
            f"IC(mean={self.mean:.4f}, std={self.std:.4f}, "
            f"IR={self.ir:.2f}, hit={self.hit_rate:.1%}, t={self.t_stat:.2f})"
        )


def ic_summary(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    method: str = "spearman",
) -> ICStats:
    """
    Compute IC summary statistics.

    Args:
        signal: Signal scores (T x N)
        returns: Forward returns (T x N)
        method: "spearman" or "pearson"

    Returns:
        ICStats dataclass
    """
    ic_series = ic(signal, returns, method)
    ic_clean = ic_series.dropna()

    if len(ic_clean) < 2:
        return ICStats(0, 0, 0, 0, 0)

    mean = ic_clean.mean()
    std = ic_clean.std()
    ir = mean / std if std > 0 else 0
    hit_rate = (ic_clean > 0).mean()
    t_stat = mean / (std / np.sqrt(len(ic_clean))) if std > 0 else 0

    return ICStats(mean=mean, std=std, ir=ir, hit_rate=hit_rate, t_stat=t_stat)


def sharpe(
    returns: Union[pd.Series, pd.DataFrame],
    rf: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Compute Sharpe Ratio.

    Args:
        returns: Return series
        rf: Annual risk-free rate
        periods_per_year: 252 for daily, 52 for weekly

    Returns:
        Annualized Sharpe ratio
    """
    if isinstance(returns, pd.DataFrame):
        returns = returns.mean(axis=1)

    returns = returns.dropna()
    if len(returns) < 2:
        return 0.0

    excess = returns - rf / periods_per_year
    mean_excess = excess.mean() * periods_per_year
    vol = returns.std() * np.sqrt(periods_per_year)

    return mean_excess / vol if vol > 0 else 0.0


def sortino(
    returns: Union[pd.Series, pd.DataFrame],
    rf: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Compute Sortino Ratio (downside risk only).

    Args:
        returns: Return series
        rf: Annual risk-free rate
        periods_per_year: 252 for daily

    Returns:
        Annualized Sortino ratio
    """
    if isinstance(returns, pd.DataFrame):
        returns = returns.mean(axis=1)

    returns = returns.dropna()
    if len(returns) < 2:
        return 0.0

    excess = returns - rf / periods_per_year
    mean_excess = excess.mean() * periods_per_year

    downside = returns[returns < 0]
    if len(downside) < 2:
        return np.inf if mean_excess > 0 else 0.0

    downside_std = downside.std() * np.sqrt(periods_per_year)
    return mean_excess / downside_std if downside_std > 0 else 0.0


def calmar(
    returns: Union[pd.Series, pd.DataFrame],
    periods_per_year: int = 252,
) -> float:
    """
    Compute Calmar Ratio (CAGR / Max Drawdown).

    Args:
        returns: Return series
        periods_per_year: 252 for daily

    Returns:
        Calmar ratio
    """
    if isinstance(returns, pd.DataFrame):
        returns = returns.mean(axis=1)

    returns = returns.dropna()
    if len(returns) < 2:
        return 0.0

    # CAGR
    total = (1 + returns).prod() - 1
    years = len(returns) / periods_per_year
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 else 0

    # Max DD
    mdd = max_drawdown(returns)

    return cagr / abs(mdd) if mdd != 0 else 0.0


def max_drawdown(
    returns: Union[pd.Series, pd.DataFrame],
) -> float:
    """
    Compute maximum drawdown.

    Args:
        returns: Return series

    Returns:
        Maximum drawdown (negative value)
    """
    if isinstance(returns, pd.DataFrame):
        returns = returns.mean(axis=1)

    returns = returns.dropna()
    if len(returns) < 2:
        return 0.0

    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak

    return dd.min()


def var(
    returns: Union[pd.Series, pd.DataFrame],
    alpha: float = 0.05,
) -> float:
    """
    Compute Value at Risk.

    Args:
        returns: Return series
        alpha: Tail probability (0.05 = 95% VaR)

    Returns:
        VaR at given alpha
    """
    if isinstance(returns, pd.DataFrame):
        returns = returns.mean(axis=1)

    returns = returns.dropna()
    return np.percentile(returns, alpha * 100)


def cvar(
    returns: Union[pd.Series, pd.DataFrame],
    alpha: float = 0.05,
) -> float:
    """
    Compute Conditional VaR (Expected Shortfall).

    Args:
        returns: Return series
        alpha: Tail probability

    Returns:
        CVaR (average of worst alpha% returns)
    """
    if isinstance(returns, pd.DataFrame):
        returns = returns.mean(axis=1)

    returns = returns.dropna()
    var_val = var(returns, alpha)
    tail = returns[returns <= var_val]

    return tail.mean() if len(tail) > 0 else var_val
