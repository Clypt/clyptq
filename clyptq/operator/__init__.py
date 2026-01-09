"""Global operators for quantitative analysis.

Base classes for custom operators:
    from clyptq.operator.base import BaseOperator, register_operator

    @register_operator("my_alpha")
    class MyAlpha(BaseOperator):
        def compute(self, data): ...

Stateless functions for quick usage:

This module provides stateless, library-level functions for:
- Time-series operations (ts_*): Rolling calculations along time axis
- Cross-sectional operations (cs_*): Operations across symbols at each timestamp
- Arithmetic operations: Basic math with proper handling of edge cases
- Comparison operators: Boolean operations for conditional logic

Usage:
    from clyptq.operator import ts_mean, ts_std, rank, normalize
    # Or via shortcut:
    from clyptq import operator

    # Time-series: compute 20-day rolling mean
    sma = ts_mean(prices, window=20)

    # Cross-sectional: rank assets at each timestamp
    ranked = rank(scores)

These operators are designed to be:
1. Stateless: No side effects, pure functions
2. Flexible: Work with both pd.Series and pd.DataFrame
3. Consistent: Same API as STAIR-RL for compatibility
4. Composable: Can be chained together for complex alpha signals
"""

# Time-series operators
from clyptq.operator.time_series import (
    # Rolling statistics
    ts_mean,
    ts_sum,
    ts_std,
    ts_min,
    ts_max,
    ts_rank,
    ts_corr,
    ts_cov,
    ts_zscore,
    ts_scale,
    ts_quantile,
    ts_product,
    ts_decayed_linear,
    # Lag/difference
    ts_delta,
    ts_delay,
    delay,
    # Argmax/min
    ts_argmax,
    ts_argmin,
    # Regression
    ts_slope,
    ts_regbeta,
    ts_residual,
    ts_rsquare,
    ts_linear_reg,
    # Days since extrema
    lowday,
    highday,
    # Returns & Cumulative
    ts_returns,
    ts_log_returns,
    ts_cumsum,
    ts_cumprod,
    # Additional Time-Series (WQ Brain compatible)
    ts_days_from_last_change,
    ts_backfill,
    ts_decay_exp_window,
    ts_hump,
    ts_jump_decay,
    ts_step_decay,
    # Missing Value Handling
    ts_fillna,
    ts_ffill,
)

# Cross-sectional operators
from clyptq.operator.cross_section import (
    rank,
    normalize,
    scale,
    scale_down,
    winsorize,
    zscore,
    demean,
    quantile_transform,
    grouped_demean,
    ScopeError,
    # New operators for STAIR-RL compatibility
    truncate,
    expand_to_global,
    signed_power,
    clip,
    softmax,
    topk_mask,
    neutralize_by_groups,
    # Portfolio weight normalization
    l1_norm,
    l2_norm,
    # Utility operators
    cs_sum,
    align_to_index,
    reindex,
    reindex_2d,
)

# Arithmetic operators
from clyptq.operator.arithmetic import (
    add,
    sub,
    mul,
    div,
    pow,
    power,
    log,
    abs,
    sqrt,
    neg,
    sign,
    min as elem_min,
    max as elem_max,
    sequence,
    twise_a_scale,
    outer_mul,
)

# Comparison operators
from clyptq.operator.comparison import (
    lt,
    gt,
    le,
    ge,
    eq,
    ne,
    or_,
    and_,
    condition,
    where,
    notna,
    isna,
    logical_and,
    logical_or,
    broadcast_mask,
)

# Base classes for custom operators
from clyptq.operator.base import (
    BaseOperator,
    OpCode,
    Expr,
    ExprDataFrame,
    wrap_operator,
    register_operator,
    get_operator,
    list_operators,
)
# Backwards compatibility
OperatorType = OpCode

# Strata-based operators (by_* functions)
from clyptq.operator.strata import (
    # Core operations
    by_demean,
    by_rank,
    by_zscore,
    by_scale,
    # Aggregation operations
    by_mean,
    by_sum,
    by_max,
    by_min,
    by_std,
    by_median,
    by_count,
    # Binning (dynamic strata)
    binning,
    binning_to_strata,
    # Convenience aliases
    sector_neutralize,
    tier_rank,
    # Additional Group Operations
    by_backfill,
    by_cartesian,
)

# Transformational and Special operators
from clyptq.operator.transform import (
    # Transformational
    trade_when,
    densify,
    # Special
    self_corr,
    vector_neut,
    purify,
    clip_extreme,
)

# Linear Algebra operators
from clyptq.operator.linalg import (
    # Basic Matrix Operations
    matmul,
    transpose,
    eye,
    diag,
    trace,
    # Decompositions
    linalg_lu,
    linalg_qr,
    linalg_svd,
    linalg_cholesky,
    linalg_eigen,
    # Matrix Properties
    linalg_det,
    linalg_rank,
    linalg_norm,
    linalg_cond,
    # Solvers
    linalg_inv,
    linalg_pinv,
    linalg_solve,
    linalg_lstsq,
    # Higher-level
    ols,
    ridge,
    # Cross-Alpha Operations
    ca_stack,
    ca_weighted_sum,
    ca_rank_average,
    ca_ic_weight,
    ca_corr,
    # Cross-Alpha Reduce Operations
    ca_reduce_avg,
    ca_reduce_sum,
    ca_reduce_max,
    ca_reduce_min,
    ca_reduce_stddev,
    ca_reduce_ir,
    ca_reduce_skewness,
    ca_reduce_kurtosis,
    ca_reduce_range,
    ca_reduce_median,
    ca_reduce_count,
    ca_reduce_norm,
    ca_reduce_powersum,
    ca_combo_a,
)

# Technical indicators (includes moving averages)
from clyptq.operator import indicator
from clyptq.operator.indicator import (
    # Moving Averages
    sma,
    ma,
    ema,
    wma,
    dema,
    tema,
    # Momentum
    rsi,
    stoch_rsi,
    macd,
    macd_line,
    macd_signal,
    macd_histogram,
    stoch,
    fast_stoch,
    slow_stoch,
    williams_r,
    cci,
    roc,
    momentum,
    # Volatility
    bollinger_bands,
    bollinger_upper,
    bollinger_lower,
    bollinger_bandwidth,
    bollinger_percent_b,
    keltner_channel,
    keltner_upper,
    keltner_lower,
    donchian_channel,
    donchian_upper,
    donchian_lower,
    donchian_middle,
    atr,
    natr,
    true_range,
    # Trend
    adx,
    adx_value,
    plus_di,
    minus_di,
    ichimoku,
    tenkan_sen,
    kijun_sen,
    senkou_span_a,
    senkou_span_b,
    chikou_span,
    supertrend,
    supertrend_value,
    supertrend_direction,
    psar,
    psar_value,
    aroon,
    aroon_up,
    aroon_down,
    aroon_oscillator,
    # Volume
    obv,
    obv_ema,
    ad,
    ad_oscillator,
    cmf,
    mfi,
    vwap,
    vwap_rolling,
    # Returns
    returns,
    log_returns,
    cumulative_returns,
    rolling_returns,
)

__all__ = [
    # Base classes
    "BaseOperator",
    "OpCode",
    "OperatorType",  # Backwards compatibility
    "Expr",
    "ExprDataFrame",
    "wrap_operator",
    "register_operator",
    "get_operator",
    "list_operators",
    # Time-series
    "ts_mean",
    "ts_sum",
    "ts_std",
    "ts_min",
    "ts_max",
    "ts_rank",
    "ts_corr",
    "ts_cov",
    "ts_zscore",
    "ts_scale",
    "ts_quantile",
    "ts_product",
    "ts_decayed_linear",
    "ts_delta",
    "ts_delay",
    "delay",
    "ts_argmax",
    "ts_argmin",
    "ts_slope",
    "ts_regbeta",
    "ts_residual",
    "ts_rsquare",
    "ts_linear_reg",
    "lowday",
    "highday",
    # Returns & Cumulative
    "ts_returns",
    "ts_log_returns",
    "ts_cumsum",
    "ts_cumprod",
    # Additional Time-Series (WQ Brain compatible)
    "ts_days_from_last_change",
    "ts_backfill",
    "ts_decay_exp_window",
    "ts_hump",
    "ts_jump_decay",
    "ts_step_decay",
    # Missing Value Handling
    "ts_fillna",
    "ts_ffill",
    # Cross-sectional
    "rank",
    "normalize",
    "scale",
    "scale_down",
    "winsorize",
    "zscore",
    "demean",
    "quantile_transform",
    "grouped_demean",
    "ScopeError",
    "truncate",
    "expand_to_global",
    "signed_power",
    "clip",
    "softmax",
    "topk_mask",
    "neutralize_by_groups",
    "l1_norm",
    "l2_norm",
    # Utility
    "cs_sum",
    "align_to_index",
    "reindex",
    "reindex_2d",
    # Arithmetic
    "add",
    "sub",
    "mul",
    "div",
    "pow",
    "power",
    "log",
    "abs",
    "sqrt",
    "neg",
    "sign",
    "elem_min",
    "elem_max",
    "sequence",
    "twise_a_scale",
    "outer_mul",
    # Comparison
    "lt",
    "gt",
    "le",
    "ge",
    "eq",
    "ne",
    "or_",
    "and_",
    "condition",
    "where",
    "notna",
    "isna",
    "logical_and",
    "logical_or",
    "broadcast_mask",
    # Technical Indicators
    "indicator",
    # Moving Averages
    "sma",
    "ma",
    "ema",
    "wma",
    "dema",
    "tema",
    # Momentum
    "rsi",
    "stoch_rsi",
    "macd",
    "macd_line",
    "macd_signal",
    "macd_histogram",
    "stoch",
    "fast_stoch",
    "slow_stoch",
    "williams_r",
    "cci",
    "roc",
    "momentum",
    # Volatility
    "bollinger_bands",
    "bollinger_upper",
    "bollinger_lower",
    "bollinger_bandwidth",
    "bollinger_percent_b",
    "keltner_channel",
    "keltner_upper",
    "keltner_lower",
    "donchian_channel",
    "donchian_upper",
    "donchian_lower",
    "donchian_middle",
    "atr",
    "natr",
    "true_range",
    # Trend
    "adx",
    "adx_value",
    "plus_di",
    "minus_di",
    "ichimoku",
    "tenkan_sen",
    "kijun_sen",
    "senkou_span_a",
    "senkou_span_b",
    "chikou_span",
    "supertrend",
    "supertrend_value",
    "supertrend_direction",
    "psar",
    "psar_value",
    "aroon",
    "aroon_up",
    "aroon_down",
    "aroon_oscillator",
    # Volume
    "obv",
    "obv_ema",
    "ad",
    "ad_oscillator",
    "cmf",
    "mfi",
    "vwap",
    "vwap_rolling",
    # Returns
    "returns",
    "log_returns",
    "cumulative_returns",
    "rolling_returns",
    # Linear Algebra - Basic
    "matmul",
    "transpose",
    "eye",
    "diag",
    "trace",
    # Linear Algebra - Decompositions
    "linalg_lu",
    "linalg_qr",
    "linalg_svd",
    "linalg_cholesky",
    "linalg_eigen",
    # Linear Algebra - Properties
    "linalg_det",
    "linalg_rank",
    "linalg_norm",
    "linalg_cond",
    # Linear Algebra - Solvers
    "linalg_inv",
    "linalg_pinv",
    "linalg_solve",
    "linalg_lstsq",
    # Linear Algebra - Higher-level
    "ols",
    "ridge",
    # Cross-Alpha Operations
    "ca_stack",
    "ca_weighted_sum",
    "ca_rank_average",
    "ca_ic_weight",
    "ca_corr",
    # Cross-Alpha Reduce Operations
    "ca_reduce_avg",
    "ca_reduce_sum",
    "ca_reduce_max",
    "ca_reduce_min",
    "ca_reduce_stddev",
    "ca_reduce_ir",
    "ca_reduce_skewness",
    "ca_reduce_kurtosis",
    "ca_reduce_range",
    "ca_reduce_median",
    "ca_reduce_count",
    "ca_reduce_norm",
    "ca_reduce_powersum",
    "ca_combo_a",
    # Strata-based operators (by_*)
    "by_demean",
    "by_rank",
    "by_zscore",
    "by_scale",
    "by_mean",
    "by_sum",
    "by_max",
    "by_min",
    "by_std",
    "by_median",
    "by_count",
    "binning",
    "binning_to_strata",
    "sector_neutralize",
    "tier_rank",
    # Additional Group Operations
    "by_backfill",
    "by_cartesian",
    # Transformational
    "trade_when",
    "densify",
    # Special
    "self_corr",
    "vector_neut",
    "purify",
    "clip_extreme",
]
