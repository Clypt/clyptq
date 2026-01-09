"""Strata-based operators (by_* functions).

Operations within stratified groups (e.g., sector, market cap tier).
These operators apply cross-sectional operations within each strata category.

Terminology:
- Strata: Categorical grouping of symbols (e.g., sector, tier)
- by_*: Operator prefix indicating strata-based operation
- binning: Dynamic strata creation from data values

Expr Support:
- All functions accept both pandas data and Expr
- When Expr is passed, returns new Expr (lazy evaluation)
- When pandas is passed, executes immediately (backward compatible)

Example:
    >>> # Get strata from provider
    >>> layer_strata = provider.get_strata("layer")
    >>> # Apply sector-neutral demean
    >>> neutralized = by_demean(alpha, strata=layer_strata)

    >>> # Dynamic strata from data
    >>> vol_bins = binning(volatility, n_bins=5)
    >>> adjusted = by_zscore(returns, strata=vol_bins)
"""

from typing import Dict, List, Optional, Union, TYPE_CHECKING
import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from clyptq.operator.base import Expr

DataType = Union[pd.Series, pd.DataFrame, "Expr"]
StrataType = Optional[Dict[str, str]]


def _is_expr(data) -> bool:
    """Check if data is an Expr instance."""
    from clyptq.operator.base import Expr
    return isinstance(data, Expr)


def _ensure_expr(data):
    """Convert data to Expr if not already."""
    from clyptq.operator.base import Expr
    if isinstance(data, Expr):
        return data
    elif isinstance(data, (int, float, np.integer, np.floating)):
        return Expr.const(float(data))
    elif isinstance(data, (pd.DataFrame, pd.Series, np.ndarray)):
        return Expr.from_data(data)
    else:
        return Expr.const(float(data))


# --- Core by_* Operators ---

def by_demean(
    data: DataType,
    strata: StrataType = None,
    *,
    lazy: bool = False,
) -> DataType:
    """Demean within each strata category.

    Subtracts the mean of each strata category from values in that category.
    Equivalent to sector neutralization when strata is sector mapping.

    Args:
        data: Input data (T x N DataFrame or Expr)
        strata: Dict mapping symbol -> category (e.g., {"BTC": "L1", "UNI": "DeFi"})
                If None, demeaning is applied across all symbols.
        lazy: If True, always return Expr

    Returns:
        Demeaned data (or Expr if input is Expr)

    Example:
        >>> # Sector-neutral alpha
        >>> layer_strata = provider.get_strata("layer")
        >>> neutral_alpha = by_demean(alpha, strata=layer_strata)
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(data) or lazy:
        data_expr = _ensure_expr(data)
        return Expr(OpCode.BY_DEMEAN, kwargs={"strata": strata}, inputs=[data_expr])

    # Immediate execution
    from clyptq.operator.base import _by_operation
    if strata is None:
        if isinstance(data, pd.DataFrame):
            return data.sub(data.mean(axis=1), axis=0)
        return data - data.mean()
    return _by_operation(data, strata, "demean")


def by_rank(
    data: DataType,
    strata: StrataType = None,
    pct: bool = True,
    *,
    lazy: bool = False,
) -> DataType:
    """Rank within each strata category.

    Computes percentile rank within each strata category.

    Args:
        data: Input data (T x N DataFrame or Expr)
        strata: Dict mapping symbol -> category
                If None, ranking is applied across all symbols.
        pct: If True, return percentile rank (0-1); else return integer rank
        lazy: If True, always return Expr

    Returns:
        Ranked data (or Expr if input is Expr)

    Example:
        >>> # Rank returns within each market cap tier
        >>> tier_strata = provider.get_strata("market_cap_tier")
        >>> ranked = by_rank(returns, strata=tier_strata)
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(data) or lazy:
        data_expr = _ensure_expr(data)
        return Expr(OpCode.BY_RANK, kwargs={"strata": strata, "pct": pct}, inputs=[data_expr])

    # Immediate execution
    from clyptq.operator.base import _by_operation
    if strata is None:
        if isinstance(data, pd.DataFrame):
            return data.rank(axis=1, pct=pct)
        return data.rank(pct=pct)
    return _by_operation(data, strata, "rank", pct=pct)


def by_zscore(
    data: DataType,
    strata: StrataType = None,
    *,
    lazy: bool = False,
) -> DataType:
    """Z-score normalize within each strata category.

    Standardizes values within each strata category to have zero mean and unit std.

    Args:
        data: Input data (T x N DataFrame or Expr)
        strata: Dict mapping symbol -> category
                If None, z-scoring is applied across all symbols.
        lazy: If True, always return Expr

    Returns:
        Z-scored data (or Expr if input is Expr)

    Example:
        >>> # Z-score within volatility tier
        >>> vol_strata = provider.get_strata("volatility_tier")
        >>> standardized = by_zscore(alpha, strata=vol_strata)
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(data) or lazy:
        data_expr = _ensure_expr(data)
        return Expr(OpCode.BY_ZSCORE, kwargs={"strata": strata}, inputs=[data_expr])

    # Immediate execution
    from clyptq.operator.base import _by_operation
    if strata is None:
        if isinstance(data, pd.DataFrame):
            mean = data.mean(axis=1)
            std = data.std(axis=1)
            return data.sub(mean, axis=0).div(std.replace(0, np.nan), axis=0)
        mean = data.mean()
        std = data.std()
        if std == 0 or pd.isna(std):
            return data * 0
        return (data - mean) / std
    return _by_operation(data, strata, "zscore")


def by_scale(
    data: DataType,
    strata: StrataType = None,
    scale_val: float = 1.0,
    *,
    lazy: bool = False,
) -> DataType:
    """Scale within each strata category.

    Normalizes values within each strata category so sum of abs values equals scale_val.

    Args:
        data: Input data (T x N DataFrame or Expr)
        strata: Dict mapping symbol -> category
                If None, scaling is applied across all symbols.
        scale_val: Target scale value (default: 1.0)
        lazy: If True, always return Expr

    Returns:
        Scaled data (or Expr if input is Expr)

    Example:
        >>> # Scale positions within each sector to sum to 1
        >>> layer_strata = provider.get_strata("layer")
        >>> scaled = by_scale(weights, strata=layer_strata, scale_val=1.0)
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(data) or lazy:
        data_expr = _ensure_expr(data)
        return Expr(OpCode.BY_SCALE, kwargs={"strata": strata, "scale_val": scale_val}, inputs=[data_expr])

    # Immediate execution
    from clyptq.operator.base import _by_operation
    if strata is None:
        if isinstance(data, pd.DataFrame):
            abs_sum = data.abs().sum(axis=1)
            return data.div(abs_sum.replace(0, np.nan), axis=0) * scale_val
        abs_sum = data.abs().sum()
        if abs_sum == 0 or pd.isna(abs_sum):
            return data * 0
        return data / abs_sum * scale_val
    return _by_operation(data, strata, "scale", scale_val=scale_val)


# --- Aggregation by_* Operators ---

def by_mean(
    data: DataType,
    strata: StrataType = None,
    *,
    lazy: bool = False,
) -> DataType:
    """Mean within each strata category.

    Computes mean of values within each strata category.
    Each symbol gets the mean of its strata.

    Args:
        data: Input data (T x N DataFrame or Expr)
        strata: Dict mapping symbol -> category
        lazy: If True, always return Expr

    Returns:
        Data with strata means (or Expr if input is Expr)
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(data) or lazy:
        data_expr = _ensure_expr(data)
        return Expr(OpCode.BY_MEAN, kwargs={"strata": strata}, inputs=[data_expr])

    from clyptq.operator.base import _by_operation
    if strata is None:
        if isinstance(data, pd.DataFrame):
            return data.mean(axis=1)
        return data.mean()
    return _by_operation(data, strata, "mean")


def by_sum(
    data: DataType,
    strata: StrataType = None,
    *,
    lazy: bool = False,
) -> DataType:
    """Sum within each strata category.

    Computes sum of values within each strata category.
    Each symbol gets the sum of its strata.

    Args:
        data: Input data (T x N DataFrame or Expr)
        strata: Dict mapping symbol -> category
        lazy: If True, always return Expr

    Returns:
        Data with strata sums (or Expr if input is Expr)
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(data) or lazy:
        data_expr = _ensure_expr(data)
        return Expr(OpCode.BY_SUM, kwargs={"strata": strata}, inputs=[data_expr])

    from clyptq.operator.base import _by_operation
    if strata is None:
        if isinstance(data, pd.DataFrame):
            return data.sum(axis=1)
        return data.sum()
    return _by_operation(data, strata, "sum")


def by_max(
    data: DataType,
    strata: StrataType = None,
    *,
    lazy: bool = False,
) -> DataType:
    """Max within each strata category.

    Computes max of values within each strata category.
    Each symbol gets the max of its strata.

    Args:
        data: Input data (T x N DataFrame or Expr)
        strata: Dict mapping symbol -> category
        lazy: If True, always return Expr

    Returns:
        Data with strata max values (or Expr if input is Expr)
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(data) or lazy:
        data_expr = _ensure_expr(data)
        return Expr(OpCode.BY_MAX, kwargs={"strata": strata}, inputs=[data_expr])

    from clyptq.operator.base import _by_operation
    if strata is None:
        if isinstance(data, pd.DataFrame):
            return data.max(axis=1)
        return data.max()
    return _by_operation(data, strata, "max")


def by_min(
    data: DataType,
    strata: StrataType = None,
    *,
    lazy: bool = False,
) -> DataType:
    """Min within each strata category.

    Computes min of values within each strata category.
    Each symbol gets the min of its strata.

    Args:
        data: Input data (T x N DataFrame or Expr)
        strata: Dict mapping symbol -> category
        lazy: If True, always return Expr

    Returns:
        Data with strata min values (or Expr if input is Expr)
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(data) or lazy:
        data_expr = _ensure_expr(data)
        return Expr(OpCode.BY_MIN, kwargs={"strata": strata}, inputs=[data_expr])

    from clyptq.operator.base import _by_operation
    if strata is None:
        if isinstance(data, pd.DataFrame):
            return data.min(axis=1)
        return data.min()
    return _by_operation(data, strata, "min")


def by_std(
    data: DataType,
    strata: StrataType = None,
    *,
    lazy: bool = False,
) -> DataType:
    """Std within each strata category.

    Computes std of values within each strata category.
    Each symbol gets the std of its strata.

    Args:
        data: Input data (T x N DataFrame or Expr)
        strata: Dict mapping symbol -> category
        lazy: If True, always return Expr

    Returns:
        Data with strata std values (or Expr if input is Expr)
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(data) or lazy:
        data_expr = _ensure_expr(data)
        return Expr(OpCode.BY_STD, kwargs={"strata": strata}, inputs=[data_expr])

    from clyptq.operator.base import _by_operation
    if strata is None:
        if isinstance(data, pd.DataFrame):
            return data.std(axis=1)
        return data.std()
    return _by_operation(data, strata, "std")


def by_median(
    data: DataType,
    strata: StrataType = None,
    *,
    lazy: bool = False,
) -> DataType:
    """Median within each strata category.

    Computes median of values within each strata category.
    Each symbol gets the median of its strata.

    Args:
        data: Input data (T x N DataFrame or Expr)
        strata: Dict mapping symbol -> category
        lazy: If True, always return Expr

    Returns:
        Data with strata median values (or Expr if input is Expr)
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(data) or lazy:
        data_expr = _ensure_expr(data)
        return Expr(OpCode.BY_MEDIAN, kwargs={"strata": strata}, inputs=[data_expr])

    from clyptq.operator.base import _by_operation
    if strata is None:
        if isinstance(data, pd.DataFrame):
            return data.median(axis=1)
        return data.median()
    return _by_operation(data, strata, "median")


def by_count(
    data: DataType,
    strata: StrataType = None,
    *,
    lazy: bool = False,
) -> DataType:
    """Count within each strata category.

    Counts non-NA values within each strata category.
    Each symbol gets the count of its strata.

    Args:
        data: Input data (T x N DataFrame or Expr)
        strata: Dict mapping symbol -> category
        lazy: If True, always return Expr

    Returns:
        Data with strata counts (or Expr if input is Expr)
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(data) or lazy:
        data_expr = _ensure_expr(data)
        return Expr(OpCode.BY_COUNT, kwargs={"strata": strata}, inputs=[data_expr])

    from clyptq.operator.base import _by_operation
    if strata is None:
        if isinstance(data, pd.DataFrame):
            return data.count(axis=1)
        return data.count()
    return _by_operation(data, strata, "count")


# --- Binning (Dynamic Strata Creation) ---

def binning(
    data: DataType,
    n_bins: int = 5,
    labels: Optional[List[str]] = None,
    *,
    lazy: bool = False,
) -> DataType:
    """Create strata from data values using quantile binning.

    Divides data into n_bins quantile bins, creating dynamic strata.
    Useful for creating strata based on data characteristics (e.g., volatility, size).

    Args:
        data: Input data (T x N DataFrame or Expr)
        n_bins: Number of bins (default: 5 for quintiles)
        labels: Optional custom labels for bins (default: ["bin_1", "bin_2", ...])
        lazy: If True, always return Expr

    Returns:
        DataFrame with bin labels as values (or Expr if input is Expr)

    Example:
        >>> # Create volatility-based strata
        >>> vol_bins = binning(volatility, n_bins=5)

        >>> # Create decile bins with custom labels
        >>> size_deciles = binning(market_cap, n_bins=10,
        ...                        labels=["D1", "D2", "D3", "D4", "D5",
        ...                                "D6", "D7", "D8", "D9", "D10"])

        >>> # Use as strata for other operations
        >>> by_demean(alpha, strata=vol_bins.iloc[-1].to_dict())
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(data) or lazy:
        data_expr = _ensure_expr(data)
        return Expr(OpCode.BINNING, kwargs={"n_bins": n_bins, "labels": labels}, inputs=[data_expr])

    # Immediate execution
    from clyptq.operator.base import _binning
    return _binning(data, n_bins=n_bins, labels=labels)


def binning_to_strata(
    binned_data: pd.DataFrame,
    timestamp_idx: int = -1,
) -> Dict[str, str]:
    """Convert binned DataFrame to strata dict at a specific timestamp.

    Helper function to convert binning output to strata format for by_* operators.

    Args:
        binned_data: DataFrame from binning() function
        timestamp_idx: Index of timestamp to use (default: -1 for last)

    Returns:
        Dict mapping symbol -> bin label

    Example:
        >>> vol_bins = binning(volatility, n_bins=5)
        >>> vol_strata = binning_to_strata(vol_bins)
        >>> adjusted = by_demean(alpha, strata=vol_strata)
    """
    if isinstance(binned_data, pd.DataFrame):
        row = binned_data.iloc[timestamp_idx]
        return {col: str(val) for col, val in row.items() if pd.notna(val)}
    elif isinstance(binned_data, pd.Series):
        return {idx: str(val) for idx, val in binned_data.items() if pd.notna(val)}
    else:
        raise TypeError(f"Expected DataFrame or Series, got {type(binned_data)}")


# --- Convenience Functions ---

def sector_neutralize(
    data: DataType,
    strata: StrataType,
    *,
    lazy: bool = False,
) -> DataType:
    """Alias for by_demean with sector strata.

    Common use case: neutralize alpha signal by sector.

    Args:
        data: Input data (alpha signal)
        strata: Sector mapping from provider.get_strata("layer") or similar
        lazy: If True, always return Expr

    Returns:
        Sector-neutralized data

    Example:
        >>> sector_strata = provider.get_strata("layer")
        >>> neutral_alpha = sector_neutralize(alpha, sector_strata)
    """
    return by_demean(data, strata=strata, lazy=lazy)


def tier_rank(
    data: DataType,
    strata: StrataType,
    pct: bool = True,
    *,
    lazy: bool = False,
) -> DataType:
    """Alias for by_rank with tier strata.

    Common use case: rank within market cap or volatility tiers.

    Args:
        data: Input data
        strata: Tier mapping from provider.get_strata("market_cap_tier") or similar
        pct: If True, return percentile rank
        lazy: If True, always return Expr

    Returns:
        Tier-ranked data

    Example:
        >>> tier_strata = provider.get_strata("market_cap_tier")
        >>> tier_ranked = tier_rank(returns, tier_strata)
    """
    return by_rank(data, strata=strata, pct=pct, lazy=lazy)


# --- Additional Group Operations (WQ Brain compatible) ---

def by_backfill(
    data: DataType,
    strata: StrataType = None,
    limit: Optional[int] = None,
    *,
    lazy: bool = False,
) -> DataType:
    """Backward fill NaN values within each strata category.

    Fills NaN values with the next valid value within each strata group.
    Useful for filling missing data while respecting group boundaries.

    Args:
        data: Input data (T x N DataFrame or Expr)
        strata: Dict mapping symbol -> category. If None, applies to all.
        limit: Maximum number of consecutive NaN to fill
        lazy: If True, always return Expr

    Returns:
        Data with NaN filled backward within each strata

    Example:
        >>> sector_strata = provider.get_strata("layer")
        >>> filled = by_backfill(sparse_data, strata=sector_strata, limit=5)
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(data) or lazy:
        data_expr = _ensure_expr(data)
        return Expr(OpCode.BY_BACKFILL, kwargs={"strata": strata, "limit": limit}, inputs=[data_expr])

    # Immediate execution
    if strata is None:
        if isinstance(data, pd.DataFrame):
            return data.bfill(limit=limit)
        return data.bfill(limit=limit)

    # Group-aware backfill
    if isinstance(data, pd.DataFrame):
        result = data.copy()
        # Group symbols by strata
        strata_groups: Dict[str, List[str]] = {}
        for symbol in data.columns:
            category = strata.get(symbol, "Unknown")
            if category not in strata_groups:
                strata_groups[category] = []
            strata_groups[category].append(symbol)

        # Backfill within each group
        for category, symbols in strata_groups.items():
            if len(symbols) > 0:
                result[symbols] = data[symbols].bfill(limit=limit)
        return result
    else:
        return data.bfill(limit=limit)


def by_cartesian(
    data1: DataType,
    data2: DataType,
    strata: StrataType = None,
    operation: str = "mul",
    *,
    lazy: bool = False,
) -> DataType:
    """Cartesian product operation within each strata category.

    Applies pairwise operation between two data sources within each strata.
    Useful for cross-symbol interactions within groups.

    Args:
        data1: First input data (T x N DataFrame or Expr)
        data2: Second input data (T x N DataFrame or Expr)
        strata: Dict mapping symbol -> category. If None, applies globally.
        operation: Operation to apply ("mul", "add", "sub", "corr")
        lazy: If True, always return Expr

    Returns:
        Result of cartesian operation within each strata

    Example:
        >>> # Compute cross-symbol correlation within sectors
        >>> sector_strata = provider.get_strata("layer")
        >>> cross_corr = by_cartesian(returns, volume, strata=sector_strata, operation="corr")
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(data1) or _is_expr(data2) or lazy:
        data1_expr = _ensure_expr(data1)
        data2_expr = _ensure_expr(data2)
        return Expr(
            OpCode.BY_CARTESIAN,
            kwargs={"strata": strata, "operation": operation},
            inputs=[data1_expr, data2_expr]
        )

    # Immediate execution
    if not isinstance(data1, pd.DataFrame) or not isinstance(data2, pd.DataFrame):
        raise ValueError("Both inputs must be DataFrames for cartesian operation")

    if strata is None:
        # Global cartesian operation
        if operation == "mul":
            return data1 * data2
        elif operation == "add":
            return data1 + data2
        elif operation == "sub":
            return data1 - data2
        elif operation == "corr":
            return data1.corrwith(data2, axis=1)
        else:
            raise ValueError(f"Unknown operation: {operation}")

    # Strata-aware cartesian operation
    result = data1.copy()
    for timestamp in data1.index:
        row1 = data1.loc[timestamp]
        row2 = data2.loc[timestamp]

        # Group symbols by strata
        strata_groups: Dict[str, List[str]] = {}
        for symbol in row1.index:
            if pd.notna(row1[symbol]):
                category = strata.get(symbol, "Unknown")
                if category not in strata_groups:
                    strata_groups[category] = []
                strata_groups[category].append(symbol)

        # Apply operation within each strata
        for category, symbols in strata_groups.items():
            if len(symbols) == 0:
                continue

            v1 = row1[symbols]
            v2 = row2[symbols]

            if operation == "mul":
                result.loc[timestamp, symbols] = v1 * v2
            elif operation == "add":
                result.loc[timestamp, symbols] = v1 + v2
            elif operation == "sub":
                result.loc[timestamp, symbols] = v1 - v2
            elif operation == "corr":
                # Cross-correlation within strata
                corr_val = v1.corr(v2)
                result.loc[timestamp, symbols] = corr_val

    return result
