"""Cross-sectional operators.

Operations across symbols at each timestamp.
For DataFrame: operates on axis=1 (across columns)
For Series: operates on entire series

Pasteurize Support (NEW DESIGN):
- in_universe: Boolean mask DataFrame/Series marking target universe membership
- scope: "GLOBAL" (compute on N) or "LOCAL" (compute on n only)
- output: "FULL" (return N) or "TRUNCATED" (return n only)

Key Principle: "Calculate on N, Constrain on n"
- GLOBAL scope + TRUNCATED output: Compute ranks on all N symbols, return only n
- LOCAL scope + TRUNCATED output: Compute ranks only on n symbols, return n
- GLOBAL scope + FULL output: Compute and return all N symbols
- LOCAL scope + FULL output: Compute on n, return n (same as LOCAL + TRUNCATED)

These are essential for alpha construction:
- rank: Convert to percentile ranks
- normalize: Mean-center (demean)
- zscore: Standardize (mean=0, std=1)
- winsorize: Handle outliers

Expr Support:
- All functions accept both pandas data and Expr
- When Expr is passed, returns new Expr (lazy evaluation)
- When pandas is passed, executes immediately (backward compatible)
"""

from typing import Union, Optional, Dict, Literal, Collection, TYPE_CHECKING
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from clyptq.operator.base import Expr

DataType = Union[pd.Series, pd.DataFrame, "Expr"]
ScopeType = Literal["GLOBAL", "LOCAL"]
OutputType = Literal["FULL", "TRUNCATED"]


class ScopeError(ValueError):
    """Error raised when invalid scope operation is attempted."""
    pass


def _is_expr(data) -> bool:
    """Check if data is an Expr instance."""
    from clyptq.operator.base import Expr
    return isinstance(data, Expr)


def _make_expr(op_code, data, *args, **kwargs):
    """Create Expr node from operation."""
    from clyptq.operator.base import Expr
    if _is_expr(data):
        return Expr(op_code, args=args, kwargs=kwargs, inputs=[data])
    # Wrap raw data and return Expr
    expr = Expr.from_data(data)
    return Expr(op_code, args=args, kwargs=kwargs, inputs=[expr])


def _get_mask(
    data: DataType,
    in_universe: Optional[DataType],
) -> Optional[DataType]:
    """Extract or create in_universe mask.

    Args:
        data: Input data
        in_universe: Explicit mask or None

    Returns:
        Boolean mask (same shape as data) or None if no mask
    """
    if in_universe is not None:
        return in_universe

    # Check if data has attached metadata
    if hasattr(data, '_in_universe'):
        return data._in_universe

    return None


def _apply_mask_for_scope(
    data: DataType,
    in_universe: Optional[DataType],
    scope: ScopeType,
) -> DataType:
    """Apply mask based on scope for computation.

    For GLOBAL scope: return data as-is (compute on all N)
    For LOCAL scope: mask out non-universe symbols (compute on n only)

    Args:
        data: Input data
        in_universe: Boolean mask
        scope: "GLOBAL" or "LOCAL"

    Returns:
        Data ready for computation
    """
    if scope == "GLOBAL" or in_universe is None:
        return data

    # LOCAL scope: set non-universe values to NaN for exclusion
    if isinstance(data, pd.DataFrame):
        # Align mask with data
        if isinstance(in_universe, pd.DataFrame):
            mask = in_universe.reindex_like(data).fillna(False)
        else:
            # Series mask (single timestamp)
            mask = pd.DataFrame(
                [in_universe.reindex(data.columns).fillna(False)] * len(data),
                index=data.index,
                columns=data.columns,
            )
        return data.where(mask)
    else:
        # Series case
        if isinstance(in_universe, pd.DataFrame):
            # Take first row if DataFrame mask
            mask = in_universe.iloc[0].reindex(data.index).fillna(False)
        else:
            mask = in_universe.reindex(data.index).fillna(False)
        return data.where(mask)


def _filter_output(
    result: DataType,
    in_universe: Optional[DataType],
    output: OutputType,
) -> DataType:
    """Filter output based on output type.

    Args:
        result: Computed result
        in_universe: Boolean mask
        output: "FULL" or "TRUNCATED"

    Returns:
        Filtered result
    """
    if output == "FULL" or in_universe is None:
        return result

    # TRUNCATED: return only universe symbols
    if isinstance(result, pd.DataFrame):
        if isinstance(in_universe, pd.DataFrame):
            # Get symbols that are ever in universe
            ever_in_universe = in_universe.any(axis=0)
            cols = [c for c in result.columns if ever_in_universe.get(c, False)]
        else:
            cols = [c for c in result.columns if in_universe.get(c, False)]
        return result[cols] if cols else result
    else:
        # Series
        if isinstance(in_universe, pd.DataFrame):
            mask = in_universe.iloc[0]
        else:
            mask = in_universe
        idxs = [i for i in result.index if mask.get(i, False)]
        return result[idxs] if idxs else result


def rank(
    data: DataType,
    in_universe: Optional[DataType] = None,
    scope: ScopeType = "GLOBAL",
    output: OutputType = "FULL",
    *,
    lazy: bool = False,
) -> DataType:
    """Cross-sectional percentile rank.

    For DataFrame: rank across columns at each row (timestamp)
    For Series: rank across all values

    Pasteurize:
    - scope="GLOBAL": Ranking computed on ALL N symbols
    - scope="LOCAL": Ranking computed only on universe (n) symbols
    - output="TRUNCATED": Filter result to universe symbols
    - output="FULL": Return all symbols (default)

    Args:
        data: Scores to rank
        in_universe: Boolean mask (T x N DataFrame or N-length Series)
        scope: "GLOBAL" (rank on N) or "LOCAL" (rank on n)
        output: "FULL" (return N) or "TRUNCATED" (return n)
        lazy: If True, always return Expr

    Returns:
        Percentile ranks between 0 and 1

    Example:
        >>> # Rank on N=500, output n=100 (Pasteurize default)
        >>> alpha = rank(close_N, in_universe=mask, scope="GLOBAL", output="TRUNCATED")

        >>> # Rank only within universe
        >>> alpha = rank(close_N, in_universe=mask, scope="LOCAL", output="TRUNCATED")
    """
    from clyptq.operator.base import OpCode

    # Expr path: build lazy expression
    if _is_expr(data) or lazy:
        return _make_expr(OpCode.CS_RANK, data, axis=1, pct=True)

    # Apply scope mask
    masked_data = _apply_mask_for_scope(data, in_universe, scope)

    if isinstance(masked_data, pd.DataFrame):
        result = masked_data.rank(axis=1, pct=True)
    else:
        result = masked_data.rank(pct=True)

    return _filter_output(result, in_universe, output)


def normalize(
    data: DataType,
    use_std: bool = False,
    in_universe: Optional[DataType] = None,
    scope: ScopeType = "GLOBAL",
    output: OutputType = "FULL",
    *,
    lazy: bool = False,
) -> DataType:
    """Cross-sectional normalization (mean removal).

    Args:
        data: Data to normalize
        use_std: If True, also divide by standard deviation
        in_universe: Boolean mask
        scope: "GLOBAL" or "LOCAL"
        output: "FULL" or "TRUNCATED"
        lazy: If True, always return Expr

    Returns:
        Mean-centered data (optionally standardized) (or Expr if input is Expr)
    """
    from clyptq.operator.base import OpCode

    # Expr path
    if _is_expr(data) or lazy:
        if use_std:
            return _make_expr(OpCode.CS_ZSCORE, data, axis=1)
        return _make_expr(OpCode.CS_DEMEAN, data, axis=1)

    masked_data = _apply_mask_for_scope(data, in_universe, scope)

    if isinstance(masked_data, pd.DataFrame):
        mean = masked_data.mean(axis=1)
        result = masked_data.sub(mean, axis=0)
        if use_std:
            std = masked_data.std(axis=1).replace(0, np.nan)
            result = result.div(std, axis=0)
    else:
        mean = masked_data.mean()
        result = masked_data - mean
        if use_std:
            std = masked_data.std()
            if std > 0:
                result = result / std

    return _filter_output(result, in_universe, output)


def demean(
    data: DataType,
    in_universe: Optional[DataType] = None,
    scope: ScopeType = "GLOBAL",
    output: OutputType = "FULL",
    *,
    lazy: bool = False,
) -> DataType:
    """Cross-sectional demeaning (subtract mean).

    Essential for market-neutral alpha signals.

    Args:
        data: Data to demean
        in_universe: Boolean mask
        scope: "GLOBAL" or "LOCAL"
        output: "FULL" or "TRUNCATED"
        lazy: If True, always return Expr

    Returns:
        Demeaned data
    """
    from clyptq.operator.base import OpCode

    # Expr path
    if _is_expr(data) or lazy:
        return _make_expr(OpCode.CS_DEMEAN, data, axis=1)

    return normalize(data, use_std=False, in_universe=in_universe, scope=scope, output=output)


def scale(
    data: DataType,
    scale_val: float = 1,
    in_universe: Optional[DataType] = None,
    scope: ScopeType = "GLOBAL",
    output: OutputType = "FULL",
    *,
    lazy: bool = False,
) -> DataType:
    """Cross-sectional scale to target absolute sum.

    Scales data so that sum of absolute values equals scale_val.

    Args:
        data: Data to scale
        scale_val: Target sum of absolute values
        in_universe: Boolean mask
        scope: "GLOBAL" or "LOCAL"
        output: "FULL" or "TRUNCATED"
        lazy: If True, always return Expr

    Returns:
        Scaled data (or Expr if input is Expr)
    """
    from clyptq.operator.base import OpCode

    # Expr path
    if _is_expr(data) or lazy:
        return _make_expr(OpCode.CS_SCALE, data, scale_val=scale_val)

    masked_data = _apply_mask_for_scope(data, in_universe, scope)

    if isinstance(masked_data, pd.DataFrame):
        abs_sum = masked_data.abs().sum(axis=1).replace(0, np.nan)
        result = masked_data.div(abs_sum, axis=0) * scale_val
    else:
        abs_sum = masked_data.abs().sum()
        if abs_sum > 0:
            result = masked_data / abs_sum * scale_val
        else:
            result = masked_data * 0

    return _filter_output(result, in_universe, output)


def scale_down(
    data: DataType,
    constant: float = 0,
    in_universe: Optional[DataType] = None,
    scope: ScopeType = "GLOBAL",
    output: OutputType = "FULL",
    *,
    lazy: bool = False,
) -> DataType:
    """Cross-sectional min-max scaling to [0, 1] range.

    Args:
        data: Data to scale
        constant: Offset to subtract from result
        in_universe: Boolean mask
        scope: "GLOBAL" or "LOCAL"
        output: "FULL" or "TRUNCATED"
        lazy: If True, always return Expr

    Returns:
        Scaled data between 0 and 1 (or Expr if input is Expr)
    """
    from clyptq.operator.base import OpCode, Expr

    # Expr path - build min-max normalization expression
    if _is_expr(data) or lazy:
        data_expr = data if _is_expr(data) else Expr.from_data(data)
        # (data - min) / (max - min) - constant
        # This is complex to express directly, use PANDAS_METHOD as fallback
        return Expr(OpCode.PANDAS_METHOD, kwargs={"method": "scale_down", "constant": constant}, inputs=[data_expr])

    masked_data = _apply_mask_for_scope(data, in_universe, scope)

    if isinstance(masked_data, pd.DataFrame):
        min_val = masked_data.min(axis=1)
        max_val = masked_data.max(axis=1)
        range_val = (max_val - min_val).replace(0, np.nan)
        result = masked_data.sub(min_val, axis=0).div(range_val, axis=0) - constant
    else:
        min_val, max_val = masked_data.min(), masked_data.max()
        range_val = max_val - min_val
        if range_val > 0:
            result = (masked_data - min_val) / range_val - constant
        else:
            result = masked_data * 0

    return _filter_output(result, in_universe, output)


def winsorize(
    data: DataType,
    std_mult: float = 4,
    in_universe: Optional[DataType] = None,
    scope: ScopeType = "GLOBAL",
    output: OutputType = "FULL",
    *,
    lazy: bool = False,
) -> DataType:
    """Cross-sectional winsorization (outlier capping).

    Caps values at mean +/- std_mult * std.

    Args:
        data: Data to winsorize
        std_mult: Number of standard deviations for cap
        in_universe: Boolean mask
        scope: "GLOBAL" or "LOCAL"
        output: "FULL" or "TRUNCATED"
        lazy: If True, always return Expr

    Returns:
        Winsorized data with extreme values capped (or Expr if input is Expr)
    """
    from clyptq.operator.base import OpCode

    # Expr path
    if _is_expr(data) or lazy:
        return _make_expr(OpCode.CS_WINSORIZE, data, std_mult=std_mult)

    masked_data = _apply_mask_for_scope(data, in_universe, scope)

    if isinstance(masked_data, pd.DataFrame):
        mean = masked_data.mean(axis=1)
        std = masked_data.std(axis=1)
        lower = mean - std_mult * std
        upper = mean + std_mult * std
        # Apply clipping row by row
        result = masked_data.copy()
        for i in range(len(masked_data)):
            result.iloc[i] = masked_data.iloc[i].clip(lower=lower.iloc[i], upper=upper.iloc[i])
    else:
        mean, std = masked_data.mean(), masked_data.std()
        result = masked_data.clip(lower=mean - std_mult * std, upper=mean + std_mult * std)

    return _filter_output(result, in_universe, output)


def zscore(
    data: DataType,
    in_universe: Optional[DataType] = None,
    scope: ScopeType = "GLOBAL",
    output: OutputType = "FULL",
    *,
    lazy: bool = False,
) -> DataType:
    """Cross-sectional z-score standardization.

    Returns (value - mean) / std at each timestamp.

    Args:
        data: Data to standardize
        in_universe: Boolean mask
        scope: "GLOBAL" or "LOCAL"
        output: "FULL" or "TRUNCATED"
        lazy: If True, always return Expr

    Returns:
        Standardized data with mean=0, std=1
    """
    from clyptq.operator.base import OpCode

    # Expr path
    if _is_expr(data) or lazy:
        return _make_expr(OpCode.CS_ZSCORE, data, axis=1)

    masked_data = _apply_mask_for_scope(data, in_universe, scope)

    if isinstance(masked_data, pd.DataFrame):
        mean = masked_data.mean(axis=1)
        std = masked_data.std(axis=1).replace(0, np.nan)
        result = masked_data.sub(mean, axis=0).div(std, axis=0)
    else:
        mean, std = masked_data.mean(), masked_data.std()
        if std > 0:
            result = (masked_data - mean) / std
        else:
            result = masked_data * 0

    return _filter_output(result, in_universe, output)


def quantile_transform(
    data: DataType,
    distribution: str = 'gaussian',
    in_universe: Optional[DataType] = None,
    scope: ScopeType = "GLOBAL",
    output: OutputType = "FULL",
    *,
    lazy: bool = False,
) -> DataType:
    """Cross-sectional quantile transformation.

    Transforms data to follow a specified distribution.

    Args:
        data: Data to transform
        distribution: 'gaussian' for normal, 'uniform' for percentile ranks
        in_universe: Boolean mask
        scope: "GLOBAL" or "LOCAL"
        output: "FULL" or "TRUNCATED"
        lazy: If True, always return Expr

    Returns:
        Transformed data (or Expr if input is Expr)
    """
    from clyptq.operator.base import OpCode

    # Expr path
    if _is_expr(data) or lazy:
        return _make_expr(OpCode.CS_QUANTILE, data, distribution=distribution)

    from scipy import stats

    masked_data = _apply_mask_for_scope(data, in_universe, scope)

    if isinstance(masked_data, pd.DataFrame):
        ranked = masked_data.rank(axis=1, pct=True)
        if distribution == 'gaussian':
            # Clip to avoid inf at 0 and 1
            result = ranked.clip(0.001, 0.999).apply(lambda x: stats.norm.ppf(x))
        else:
            result = ranked
    else:
        ranked = masked_data.rank(pct=True).clip(0.001, 0.999)
        if distribution == 'gaussian':
            result = pd.Series(stats.norm.ppf(ranked.values), index=masked_data.index)
        else:
            result = ranked

    return _filter_output(result, in_universe, output)


def grouped_demean(
    data: DataType,
    groups: Optional[Dict[str, str]] = None,
    in_universe: Optional[DataType] = None,
    scope: ScopeType = "GLOBAL",
    output: OutputType = "FULL",
    *,
    lazy: bool = False,
) -> DataType:
    """Group-based demeaning (sector neutralization).

    Args:
        data: Data to demean
        groups: Mapping of {symbol: group_name}. If None, uses overall mean.
        in_universe: Boolean mask
        scope: "GLOBAL" or "LOCAL"
        output: "FULL" or "TRUNCATED"
        lazy: Force lazy evaluation (return Expr)

    Returns:
        Data with group means removed

    Example:
        >>> groups = {'AAPL': 'Tech', 'MSFT': 'Tech', 'JPM': 'Finance'}
        >>> grouped_demean(scores, groups, in_universe=mask, scope="LOCAL")
        # Tech stocks demeaned within Tech
        # Finance stocks demeaned within Finance
    """
    from clyptq.operator.base import OpCode

    if _is_expr(data) or lazy:
        return _make_expr(OpCode.CS_GROUPED_DEMEAN, data, groups=groups)

    if groups is None:
        return demean(data, in_universe=in_universe, scope=scope, output=output)

    masked_data = _apply_mask_for_scope(data, in_universe, scope)

    if isinstance(masked_data, pd.DataFrame):
        result = masked_data.copy()
        for timestamp in masked_data.index:
            row = masked_data.loc[timestamp]
            # Group by sector
            sector_groups: Dict[str, list] = {}
            for symbol in row.index:
                if pd.notna(row[symbol]):  # Only include non-NaN values
                    sector = groups.get(symbol, "Unknown")
                    if sector not in sector_groups:
                        sector_groups[sector] = []
                    sector_groups[sector].append(symbol)

            # Demean within each sector
            for sector, symbols in sector_groups.items():
                sector_values = row[symbols]
                sector_mean = sector_values.mean()
                result.loc[timestamp, symbols] = sector_values - sector_mean
    else:
        # Series case
        sector_groups: Dict[str, list] = {}
        for symbol, value in masked_data.items():
            if pd.notna(value):
                sector = groups.get(symbol, "Unknown")
                if sector not in sector_groups:
                    sector_groups[sector] = []
                sector_groups[sector].append(symbol)

        result = masked_data.copy()
        for sector, symbols in sector_groups.items():
            sector_values = masked_data[symbols]
            sector_mean = sector_values.mean()
            result[symbols] = sector_values - sector_mean

    return _filter_output(result, in_universe, output)


# --- Additional operators for STAIR-RL compatibility ---

def truncate(
    data: DataType,
    in_universe: DataType,
) -> DataType:
    """Filter to universe symbols only (convenience function).

    Equivalent to: _filter_output(data, in_universe, "TRUNCATED")

    Args:
        data: Input data
        in_universe: Boolean mask

    Returns:
        Data with only universe symbols
    """
    return _filter_output(data, in_universe, "TRUNCATED")


def expand_to_global(
    data: DataType,
    all_symbols: Collection[str],
    fill_value: float = np.nan,
) -> DataType:
    """Expand truncated data to global scope.

    Args:
        data: Truncated data (n symbols)
        all_symbols: All symbols (N)
        fill_value: Value for symbols not in data

    Returns:
        Data with N symbols
    """
    if isinstance(data, pd.DataFrame):
        return data.reindex(columns=sorted(all_symbols), fill_value=fill_value)
    else:
        return data.reindex(sorted(all_symbols), fill_value=fill_value)


def signed_power(
    data: DataType,
    power: float,
    in_universe: Optional[DataType] = None,
    scope: ScopeType = "GLOBAL",
    output: OutputType = "FULL",
) -> DataType:
    """Apply signed power transformation.

    sign(x) * |x|^power

    Useful for amplifying strong signals while preserving sign.

    Args:
        data: Data to transform
        power: Power to apply (e.g., 0.5 for sqrt, 2 for square)
        in_universe: Boolean mask
        scope: "GLOBAL" or "LOCAL"
        output: "FULL" or "TRUNCATED"

    Returns:
        Transformed data
    """
    masked_data = _apply_mask_for_scope(data, in_universe, scope)
    result = np.sign(masked_data) * np.abs(masked_data) ** power
    return _filter_output(result, in_universe, output)


def clip(
    data: DataType,
    lower: Optional[float] = None,
    upper: Optional[float] = None,
    in_universe: Optional[DataType] = None,
    scope: ScopeType = "GLOBAL",
    output: OutputType = "FULL",
    *,
    lazy: bool = False,
) -> DataType:
    """Clip values to specified range.

    Args:
        data: Data to clip
        lower: Minimum value
        upper: Maximum value
        in_universe: Boolean mask
        scope: "GLOBAL" or "LOCAL"
        output: "FULL" or "TRUNCATED"
        lazy: If True, always return Expr

    Returns:
        Clipped data (or Expr if input is Expr)
    """
    from clyptq.operator.base import OpCode

    # Expr path
    if _is_expr(data) or lazy:
        return _make_expr(OpCode.CS_CLIP, data, lower=lower, upper=upper)

    masked_data = _apply_mask_for_scope(data, in_universe, scope)
    result = masked_data.clip(lower=lower, upper=upper)
    return _filter_output(result, in_universe, output)


def softmax(
    data: DataType,
    temperature: float = 1.0,
    in_universe: Optional[DataType] = None,
    scope: ScopeType = "GLOBAL",
    output: OutputType = "FULL",
    *,
    lazy: bool = False,
) -> DataType:
    """Cross-sectional softmax.

    Converts scores to probabilities that sum to 1.

    Args:
        data: Data to transform
        temperature: Softmax temperature (higher = more uniform)
        in_universe: Boolean mask
        scope: "GLOBAL" or "LOCAL"
        output: "FULL" or "TRUNCATED"
        lazy: If True, always return Expr

    Returns:
        Softmax probabilities (or Expr if input is Expr)
    """
    from clyptq.operator.base import OpCode

    # Expr path
    if _is_expr(data) or lazy:
        return _make_expr(OpCode.CS_SOFTMAX, data, temperature=temperature)

    masked_data = _apply_mask_for_scope(data, in_universe, scope)

    if isinstance(masked_data, pd.DataFrame):
        # Subtract max for numerical stability
        shifted = masked_data.sub(masked_data.max(axis=1), axis=0) / temperature
        exp_data = np.exp(shifted)
        result = exp_data.div(exp_data.sum(axis=1), axis=0)
    else:
        shifted = (masked_data - masked_data.max()) / temperature
        exp_data = np.exp(shifted)
        result = exp_data / exp_data.sum()

    return _filter_output(result, in_universe, output)


def topk_mask(
    data: DataType,
    k: int,
    ascending: bool = False,
    in_universe: Optional[DataType] = None,
    scope: ScopeType = "GLOBAL",
    *,
    lazy: bool = False,
) -> DataType:
    """Create mask for top-k values.

    Args:
        data: Data to rank
        k: Number of top values
        ascending: If True, select bottom-k instead
        in_universe: Boolean mask
        scope: "GLOBAL" or "LOCAL"
        lazy: If True, always return Expr

    Returns:
        Boolean mask where True indicates top-k (or Expr if input is Expr)
    """
    from clyptq.operator.base import OpCode

    # Expr path
    if _is_expr(data) or lazy:
        return _make_expr(OpCode.CS_TOPK_MASK, data, k=k, ascending=ascending)

    masked_data = _apply_mask_for_scope(data, in_universe, scope)

    if isinstance(masked_data, pd.DataFrame):
        ranks = masked_data.rank(axis=1, ascending=ascending)
        result = ranks <= k
    else:
        ranks = masked_data.rank(ascending=ascending)
        result = ranks <= k

    return result


def neutralize_by_groups(
    data: DataType,
    group_values: DataType,
    n_groups: int = 5,
    in_universe: Optional[DataType] = None,
    scope: ScopeType = "GLOBAL",
    output: OutputType = "FULL",
) -> DataType:
    """Neutralize by quantile groups of another variable.

    Demean within quantile bins of group_values (e.g., market cap quintiles).

    Args:
        data: Data to neutralize
        group_values: Values to create groups from (e.g., market cap)
        n_groups: Number of quantile groups
        in_universe: Boolean mask
        scope: "GLOBAL" or "LOCAL"
        output: "FULL" or "TRUNCATED"

    Returns:
        Data neutralized within each quantile group
    """
    masked_data = _apply_mask_for_scope(data, in_universe, scope)
    masked_groups = _apply_mask_for_scope(group_values, in_universe, scope)

    if isinstance(masked_data, pd.DataFrame):
        result = masked_data.copy()
        for timestamp in masked_data.index:
            row_data = masked_data.loc[timestamp].dropna()
            row_groups = masked_groups.loc[timestamp].dropna()

            # Get common symbols
            common = row_data.index.intersection(row_groups.index)
            if len(common) == 0:
                continue

            row_data = row_data[common]
            row_groups = row_groups[common]

            # Create quantile groups
            try:
                bins = pd.qcut(row_groups, n_groups, labels=False, duplicates='drop')
            except ValueError:
                # Not enough unique values
                continue

            # Demean within each group
            for group_id in bins.unique():
                group_mask = bins == group_id
                symbols = bins[group_mask].index
                group_mean = row_data[symbols].mean()
                result.loc[timestamp, symbols] = row_data[symbols] - group_mean
    else:
        # Series case
        row_data = masked_data.dropna()
        row_groups = masked_groups.dropna()
        common = row_data.index.intersection(row_groups.index)

        if len(common) > 0:
            row_data = row_data[common]
            row_groups = row_groups[common]

            try:
                bins = pd.qcut(row_groups, n_groups, labels=False, duplicates='drop')
                result = row_data.copy()
                for group_id in bins.unique():
                    group_mask = bins == group_id
                    symbols = bins[group_mask].index
                    group_mean = row_data[symbols].mean()
                    result[symbols] = row_data[symbols] - group_mean
            except ValueError:
                result = masked_data
        else:
            result = masked_data

    return _filter_output(result, in_universe, output)


# --- Normalization operators for portfolio weights ---

def l1_norm(
    data: DataType,
    in_universe: Optional[DataType] = None,
    scope: ScopeType = "GLOBAL",
    output: OutputType = "FULL",
    *,
    lazy: bool = False,
) -> DataType:
    """L1 normalize: sum(|w|) = 1.

    Essential for portfolio weight generation from alpha signals.
    After L1 normalization, weights can be directly multiplied by book size
    to get target positions.

    Args:
        data: Data to normalize
        in_universe: Boolean mask
        scope: "GLOBAL" or "LOCAL"
        output: "FULL" or "TRUNCATED"
        lazy: If True, always return Expr

    Returns:
        L1 normalized data where sum(|values|) = 1

    Example:
        >>> alpha = demean(raw_scores)  # Zero mean
        >>> weights = l1_norm(alpha)     # Sum(|w|) = 1
        >>> target_positions = weights * book_size
    """
    from clyptq.operator.base import OpCode

    # Expr path
    if _is_expr(data) or lazy:
        return _make_expr(OpCode.CS_L1_NORM, data, axis=1)

    masked_data = _apply_mask_for_scope(data, in_universe, scope)

    if isinstance(masked_data, pd.DataFrame):
        abs_sum = masked_data.abs().sum(axis=1).replace(0, np.nan)
        result = masked_data.div(abs_sum, axis=0)
    else:
        abs_sum = masked_data.abs().sum()
        if abs_sum > 0:
            result = masked_data / abs_sum
        else:
            result = masked_data * 0

    return _filter_output(result, in_universe, output)


def l2_norm(
    data: DataType,
    in_universe: Optional[DataType] = None,
    scope: ScopeType = "GLOBAL",
    output: OutputType = "FULL",
    *,
    lazy: bool = False,
) -> DataType:
    """L2 normalize: sqrt(sum(w^2)) = 1.

    Alternative normalization that penalizes concentration.
    L2 normalized weights have unit Euclidean length.

    Args:
        data: Data to normalize
        in_universe: Boolean mask
        scope: "GLOBAL" or "LOCAL"
        output: "FULL" or "TRUNCATED"
        lazy: If True, always return Expr

    Returns:
        L2 normalized data where sqrt(sum(values^2)) = 1

    Example:
        >>> alpha = demean(raw_scores)
        >>> weights = l2_norm(alpha)  # Unit length vector
    """
    from clyptq.operator.base import OpCode

    # Expr path
    if _is_expr(data) or lazy:
        return _make_expr(OpCode.CS_L2_NORM, data, axis=1)

    masked_data = _apply_mask_for_scope(data, in_universe, scope)

    if isinstance(masked_data, pd.DataFrame):
        l2_sum = np.sqrt((masked_data ** 2).sum(axis=1)).replace(0, np.nan)
        result = masked_data.div(l2_sum, axis=0)
    else:
        l2_sum = np.sqrt((masked_data ** 2).sum())
        if l2_sum > 0:
            result = masked_data / l2_sum
        else:
            result = masked_data * 0

    return _filter_output(result, in_universe, output)


# --- Utility operators for alignment ---

def cs_sum(
    data: DataType,
    in_universe: Optional[DataType] = None,
    scope: ScopeType = "GLOBAL",
    *,
    lazy: bool = False,
) -> DataType:
    """Cross-sectional sum."""
    from clyptq.operator.base import OpCode

    if _is_expr(data) or lazy:
        return _make_expr(OpCode.CS_SUM, data)

    masked_data = _apply_mask_for_scope(data, in_universe, scope)

    if isinstance(masked_data, pd.DataFrame):
        return masked_data.sum(axis=1)
    else:
        return masked_data.sum()


def align_to_index(
    mapping: Union[Dict, pd.Series],
    data: DataType,
    default=None,
    *,
    lazy: bool = False,
) -> DataType:
    """Align dict/Series to data's index."""
    from clyptq.operator.base import OpCode

    if _is_expr(data) or lazy:
        return _make_expr(OpCode.ALIGN_TO_INDEX, data, mapping=mapping, default=default)

    if isinstance(mapping, dict):
        mapping = pd.Series(mapping)

    if isinstance(data, pd.DataFrame):
        index = data.columns
    else:
        index = data.index

    result = mapping.reindex(index)
    if default is not None:
        result = result.fillna(default)

    return result


def reindex(
    source: DataType,
    target: DataType,
    fill_value=None,
    *,
    lazy: bool = False,
) -> DataType:
    """Reindex source to match target's index."""
    from clyptq.operator.base import OpCode

    if _is_expr(source) or _is_expr(target) or lazy:
        return _make_expr(OpCode.REINDEX, source, target=target, fill_value=fill_value)

    if isinstance(target, pd.DataFrame):
        index = target.columns
    else:
        index = target.index

    if isinstance(source, pd.DataFrame):
        result = source.reindex(columns=index)
    else:
        result = source.reindex(index)

    if fill_value is not None:
        result = result.fillna(fill_value)

    return result


def reindex_2d(
    source: DataType,
    target: DataType,
    fill_value=None,
    *,
    lazy: bool = False,
) -> DataType:
    """Reindex 2D matrix to match target's index on both axes."""
    from clyptq.operator.base import OpCode

    if _is_expr(source) or _is_expr(target) or lazy:
        return _make_expr(OpCode.REINDEX_2D, source, target=target, fill_value=fill_value)

    if isinstance(target, pd.DataFrame):
        index = target.columns
    else:
        index = target.index

    result = source.reindex(index=index, columns=index)

    if fill_value is not None:
        result = result.fillna(fill_value)

    return result


