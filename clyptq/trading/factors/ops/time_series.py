"""
Time-series operations for processing sequential data.
"""

import numpy as np


def ts_mean(series: np.ndarray, period: int) -> float:
    """
    Time-series mean over period.

    Args:
        series: Price or value series
        period: Lookback period

    Returns:
        Mean of last period values
    """
    if len(series) < period:
        return np.nan

    return float(np.mean(series[-period:]))


def ts_std(series: np.ndarray, period: int) -> float:
    """
    Time-series standard deviation over period.

    Args:
        series: Price or value series
        period: Lookback period

    Returns:
        Std of last period values
    """
    if len(series) < period:
        return np.nan

    return float(np.std(series[-period:]))


def ts_sum(series: np.ndarray, period: int) -> float:
    """
    Time-series sum over period.

    Args:
        series: Price or value series
        period: Lookback period

    Returns:
        Sum of last period values
    """
    if len(series) < period:
        return np.nan

    return float(np.sum(series[-period:]))


def ts_min(series: np.ndarray, period: int) -> float:
    """
    Time-series minimum over period.

    Args:
        series: Price or value series
        period: Lookback period

    Returns:
        Min of last period values
    """
    if len(series) < period:
        return np.nan

    return float(np.min(series[-period:]))


def ts_max(series: np.ndarray, period: int) -> float:
    """
    Time-series maximum over period.

    Args:
        series: Price or value series
        period: Lookback period

    Returns:
        Max of last period values
    """
    if len(series) < period:
        return np.nan

    return float(np.max(series[-period:]))


def delay(series: np.ndarray, period: int) -> float:
    """
    Get value from period bars ago.

    Args:
        series: Price or value series
        period: Lookback period

    Returns:
        Value at series[-period]
    """
    if len(series) <= period:
        return np.nan

    return float(series[-(period + 1)])


def delta(series: np.ndarray, period: int) -> float:
    """
    Change over period: current - delay(period).

    Args:
        series: Price or value series
        period: Lookback period

    Returns:
        series[-1] - series[-(period+1)]
    """
    if len(series) <= period:
        return np.nan

    return float(series[-1] - series[-(period + 1)])


def ts_rank(series: np.ndarray, period: int) -> float:
    """
    Percentile rank of current value in period, normalized to [0, 1].

    Args:
        series: Price or value series
        period: Lookback period

    Returns:
        Rank of current value in [0, 1]
    """
    if len(series) < period:
        return np.nan

    window = series[-period:]
    current = window[-1]

    rank = (window < current).sum() / len(window)

    return float(rank)


def correlation(series_x: np.ndarray, series_y: np.ndarray, period: int) -> float:
    """
    Rolling correlation between two series.

    Args:
        series_x: First series
        series_y: Second series
        period: Lookback period

    Returns:
        Correlation coefficient
    """
    if len(series_x) < period or len(series_y) < period:
        return np.nan

    x = series_x[-period:]
    y = series_y[-period:]

    if len(x) != len(y):
        return np.nan

    corr = np.corrcoef(x, y)[0, 1]

    return float(corr) if not np.isnan(corr) else 0.0
