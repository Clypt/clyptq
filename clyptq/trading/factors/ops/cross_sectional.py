"""
Cross-sectional operations for comparing values across symbols at a point in time.
"""

from typing import Dict

import numpy as np


def rank(values: Dict[str, float]) -> Dict[str, float]:
    """
    Rank values across symbols, normalized to [0, 1].

    Args:
        values: {symbol: value}

    Returns:
        {symbol: rank} where rank is in [0, 1]
    """
    if not values:
        return {}

    symbols = list(values.keys())
    vals = np.array([values[s] for s in symbols])

    ranks = np.argsort(np.argsort(vals))
    n = len(ranks)

    if n == 1:
        normalized = np.array([0.5])
    else:
        normalized = ranks / (n - 1)

    return {symbols[i]: float(normalized[i]) for i in range(len(symbols))}


def normalize(values: Dict[str, float]) -> Dict[str, float]:
    """
    Z-score normalization across symbols.

    Args:
        values: {symbol: value}

    Returns:
        {symbol: normalized_value}
    """
    if not values:
        return {}

    symbols = list(values.keys())
    vals = np.array([values[s] for s in symbols])

    mean = np.mean(vals)
    std = np.std(vals)

    if std < 1e-10:
        return {s: 0.0 for s in symbols}

    normalized = (vals - mean) / std

    return {symbols[i]: float(normalized[i]) for i in range(len(symbols))}


def winsorize(values: Dict[str, float], lower: float = 0.05, upper: float = 0.95) -> Dict[str, float]:
    """
    Winsorize values by capping at percentiles.

    Args:
        values: {symbol: value}
        lower: Lower percentile (0-1)
        upper: Upper percentile (0-1)

    Returns:
        {symbol: winsorized_value}
    """
    if not values:
        return {}

    symbols = list(values.keys())
    vals = np.array([values[s] for s in symbols])

    lower_bound = np.percentile(vals, lower * 100)
    upper_bound = np.percentile(vals, upper * 100)

    winsorized = np.clip(vals, lower_bound, upper_bound)

    return {symbols[i]: float(winsorized[i]) for i in range(len(symbols))}


def demean(values: Dict[str, float]) -> Dict[str, float]:
    """
    Remove cross-sectional mean.

    Args:
        values: {symbol: value}

    Returns:
        {symbol: demeaned_value}
    """
    if not values:
        return {}

    symbols = list(values.keys())
    vals = np.array([values[s] for s in symbols])

    mean = np.mean(vals)
    demeaned = vals - mean

    return {symbols[i]: float(demeaned[i]) for i in range(len(symbols))}
