"""Base classes for operators with lazy evaluation support.

Core Design:
1. All operators return Expr
2. Operations between Expr create new Expr (AST construction)
3. At execution time, evaluate with pandas or convert to Rust

Usage:
    # Function style (internally returns Expr)
    signal = ts_mean(close, 20)  # Expr
    signal = rank(signal)         # Expr
    signal = signal * 2 + 1       # Expr (operator overloading)

    # Execute
    result = signal.execute()     # pandas DataFrame

    # Rust conversion (future)
    spec = signal.to_spec()       # JSON-serializable dict
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from functools import wraps

import numpy as np
import pandas as pd


class OpCode(Enum):
    """Operation codes for Rust backend mapping."""

    # Data source
    COLUMN = "column"
    CONSTANT = "constant"

    # Time-series (rolling)
    TS_MEAN = "ts_mean"
    TS_STD = "ts_std"
    TS_SUM = "ts_sum"
    TS_MIN = "ts_min"
    TS_MAX = "ts_max"
    TS_RANK = "ts_rank"
    TS_DELTA = "ts_delta"
    TS_DELAY = "ts_delay"
    TS_SLOPE = "ts_slope"
    TS_ZSCORE = "ts_zscore"
    TS_CORR = "ts_corr"
    TS_COV = "ts_cov"
    TS_DECAYED_LINEAR = "ts_decayed_linear"
    TS_ARGMAX = "ts_argmax"
    TS_ARGMIN = "ts_argmin"
    TS_QUANTILE = "ts_quantile"
    TS_PRODUCT = "ts_product"
    TS_SCALE = "ts_scale"
    LOWDAY = "lowday"
    HIGHDAY = "highday"
    TS_REGBETA = "ts_regbeta"
    TS_RESIDUAL = "ts_residual"
    TS_RSQUARE = "ts_rsquare"
    TS_LINEAR_REG = "ts_linear_reg"
    TS_RETURNS = "ts_returns"       # pct_change
    TS_LOG_RETURNS = "ts_log_returns"  # log returns
    TS_CUMSUM = "ts_cumsum"         # cumulative sum
    TS_CUMPROD = "ts_cumprod"       # cumulative product

    # Time-series additional (WQ Brain)
    TS_DAYS_FROM_LAST_CHANGE = "ts_days_from_last_change"  # days since last change
    TS_BACKFILL = "ts_backfill"     # backfill missing values
    TS_COUNT_NANS = "ts_count_nans"  # count of NaN values
    TS_AV_DIFF = "ts_av_diff"       # mean difference ignoring NaN
    TS_STEP = "ts_step"             # day counter
    TS_HUMP = "ts_hump"             # turnover limit
    TS_JUMP_DECAY = "ts_jump_decay"  # jump detection and decay
    TS_KTH_ELEMENT = "ts_kth_element"  # k-th valid value
    TS_LAST_DIFF_VALUE = "ts_last_diff_value"  # last different value

    # Cross-sectional (per row)
    CS_RANK = "cs_rank"
    CS_DEMEAN = "cs_demean"
    CS_ZSCORE = "cs_zscore"
    CS_WINSORIZE = "cs_winsorize"
    CS_L1_NORM = "cs_l1_norm"
    CS_L2_NORM = "cs_l2_norm"
    CS_SCALE = "cs_scale"
    CS_SOFTMAX = "cs_softmax"
    CS_CLIP = "cs_clip"
    CS_TOPK_MASK = "cs_topk_mask"
    CS_QUANTILE = "cs_quantile"
    CS_GROUPED_DEMEAN = "cs_grouped_demean"  # sector neutralization
    CS_SUM = "cs_sum"                        # cross-sectional sum
    ALIGN_TO_INDEX = "align_to_index"
    REINDEX_2D = "reindex_2d"

    # Strata-based operations (by_* operators)
    # These operate within each strata category (e.g., within each sector)
    BY_DEMEAN = "by_demean"          # demean within each strata
    BY_RANK = "by_rank"              # rank within each strata
    BY_ZSCORE = "by_zscore"          # z-score within each strata
    BY_SCALE = "by_scale"            # scale within each strata
    BY_MEAN = "by_mean"              # mean within each strata
    BY_SUM = "by_sum"                # sum within each strata
    BY_MAX = "by_max"                # max within each strata
    BY_MIN = "by_min"                # min within each strata
    BY_STD = "by_std"                # std within each strata
    BY_MEDIAN = "by_median"          # median within each strata
    BY_COUNT = "by_count"            # count within each strata

    # Binning (dynamic strata creation)
    BINNING = "binning"              # create strata from data values

    # Strata additional operations (WQ Brain: group_*)
    BY_BACKFILL = "by_backfill"     # backfill within group
    BY_CARTESIAN = "by_cartesian"   # group cross product

    # Transformational (WQ Brain)
    TRADE_WHEN = "trade_when"       # conditional trade/hold
    DENSIFY = "densify"             # group compression

    # Special Operations (WQ Brain)
    SELF_CORR = "self_corr"         # auto-correlation matrix (D x N) -> (D x N x N)
    VECTOR_NEUT = "vector_neut"     # vector orthogonalization
    CLIP_EXTREME = "clip_extreme"   # extreme value clipping
    PURIFY = "purify"               # multi-factor neutralization

    # Arithmetic (element-wise)
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    POW = "pow"
    LOG = "log"
    LOG10 = "log10"
    LOG2 = "log2"
    SQRT = "sqrt"
    ABS = "abs"
    NEG = "neg"
    SIGN = "sign"
    EXP = "exp"
    FLOOR = "floor"
    CEIL = "ceil"
    ROUND = "round"
    OUTER_MUL = "outer_mul"  # outer product: row_series(T,) x col_series(N,) -> DataFrame(T x N)

    # Reduction
    SUM = "sum"
    MEAN = "mean"
    STD = "std"
    VAR = "var"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"
    PROD = "prod"
    COUNT = "count"
    CUMSUM = "cumsum"
    CUMPROD = "cumprod"
    CUMMAX = "cummax"
    CUMMIN = "cummin"

    # Comparison
    LT = "lt"
    GT = "gt"
    LE = "le"
    GE = "ge"
    EQ = "eq"
    NE = "ne"
    AND = "and"
    OR = "or"
    NOT = "not"
    CONDITION = "condition"  # where/if-then-else

    # DataFrame operations
    FILLNA = "fillna"
    DROPNA = "dropna"
    REPLACE = "replace"
    REINDEX = "reindex"
    FFILL = "ffill"
    BFILL = "bfill"
    INTERPOLATE = "interpolate"
    QUANTILE = "quantile"

    # Element-wise binary
    ELEM_MIN = "elem_min"  # np.minimum
    ELEM_MAX = "elem_max"  # np.maximum

    # Technical indicators
    SMA = "sma"
    EMA = "ema"
    WMA = "wma"
    DEMA = "dema"
    TEMA = "tema"
    RSI = "rsi"
    MACD = "macd"
    BOLLINGER = "bollinger"
    ATR = "atr"
    WILLIAMS_R = "williams_r"
    CCI = "cci"
    MFI = "mfi"
    STOCH = "stoch"
    ADX = "adx"
    OBV = "obv"
    VWAP = "vwap"

    # Linear Algebra - Basic Matrix Operations
    MATMUL = "matmul"              # A @ B (matrix multiplication)
    TRANSPOSE = "transpose"        # A.T (transpose)
    EYE = "eye"                    # identity matrix creation
    DIAG = "diag"                  # diagonal matrix creation/extraction
    TRACE = "trace"                # trace (sum of diagonal)

    # Linear Algebra - Decompositions
    LINALG_LU = "linalg_lu"        # LU decomposition
    LINALG_QR = "linalg_qr"        # QR decomposition
    LINALG_SVD = "linalg_svd"      # SVD decomposition
    LINALG_CHOLESKY = "linalg_cholesky"  # Cholesky decomposition
    LINALG_EIGEN = "linalg_eigen"  # eigenvalue decomposition

    # Linear Algebra - Matrix Properties
    LINALG_DET = "linalg_det"      # determinant
    LINALG_RANK = "linalg_rank"    # matrix rank
    LINALG_NORM = "linalg_norm"    # matrix/vector norm
    LINALG_COND = "linalg_cond"    # condition number

    # Linear Algebra - Solvers
    LINALG_INV = "linalg_inv"      # inverse matrix
    LINALG_PINV = "linalg_pinv"    # pseudo-inverse
    LINALG_SOLVE = "linalg_solve"  # solve Ax = b
    LINALG_LSTSQ = "linalg_lstsq"  # least squares solution (OLS)

    # Cross-Alpha Operations (for Combiner)
    CA_STACK = "ca_stack"          # alpha stack (multiple alphas to tensor)
    CA_WEIGHTED_SUM = "ca_weighted_sum"    # weighted average
    CA_RANK_AVERAGE = "ca_rank_average"    # rank average
    CA_IC_WEIGHT = "ca_ic_weight"          # IC-based dynamic weighting
    CA_CORR = "ca_corr"            # inter-alpha correlation matrix

    # Cross-Alpha Reduce Operations (WQ Brain: reduce_*)
    CA_REDUCE_AVG = "ca_reduce_avg"        # alpha mean
    CA_REDUCE_SUM = "ca_reduce_sum"        # alpha sum
    CA_REDUCE_MAX = "ca_reduce_max"        # alpha maximum
    CA_REDUCE_MIN = "ca_reduce_min"        # alpha minimum
    CA_REDUCE_STDDEV = "ca_reduce_stddev"  # alpha standard deviation
    CA_REDUCE_IR = "ca_reduce_ir"          # Information Ratio
    CA_REDUCE_SKEWNESS = "ca_reduce_skewness"  # skewness
    CA_REDUCE_KURTOSIS = "ca_reduce_kurtosis"  # kurtosis
    CA_REDUCE_RANGE = "ca_reduce_range"    # range (max - min)
    CA_REDUCE_MEDIAN = "ca_reduce_median"  # median
    CA_REDUCE_COUNT = "ca_reduce_count"    # valid value count
    CA_REDUCE_NORM = "ca_reduce_norm"      # L1 norm (sum of abs)
    CA_REDUCE_POWERSUM = "ca_reduce_powersum"  # power sum

    # Combo Operations (WQ Brain: combo_*)
    CA_COMBO_A = "ca_combo_a"      # alpha combination (IR-based weighting)

    # Custom/generic
    CUSTOM = "custom"
    PANDAS_METHOD = "pandas_method"
    NUMPY_FUNC = "numpy_func"
    APPLY = "apply"  # generic apply


class Expr:
    """Lazy expression node - a node in the expression tree.

    All operator operations return Expr, and operations between Expr also return Expr.
    This records the entire operation as an AST, which can be executed at once or converted to Rust.

    Supports:
    - All arithmetic operations (+, -, *, /, **, etc.)
    - All comparison operations (<, >, ==, etc.)
    - Pandas methods (.rolling(), .shift(), .fillna(), etc.)
    - Numpy ufuncs (np.sqrt(), np.log(), np.abs(), etc.)

    Example:
        ```python
        # Build expression tree (not executed yet)
        close = Expr.col("close")
        signal = close.rolling(20).mean()  # Expr
        signal = signal.rank(axis=1)        # Expr
        signal = signal * 2 - 1             # Expr (operator overloading)

        # Execute
        result = signal.execute(data)  # pandas DataFrame
        ```
    """

    def __init__(
        self,
        op: OpCode,
        args: Tuple[Any, ...] = (),
        kwargs: Optional[Dict[str, Any]] = None,
        inputs: Optional[List["Expr"]] = None,
    ):
        """Initialize expression node.

        Args:
            op: Operation code
            args: Positional arguments for the operation
            kwargs: Keyword arguments for the operation
            inputs: Input expressions (children in the tree)
        """
        self.op = op
        self.args = args
        self.kwargs = kwargs or {}
        self.inputs = inputs or []

    # --- Factory Methods ---

    @classmethod
    def col(cls, name: str) -> "Expr":
        """Create column reference expression."""
        return cls(OpCode.COLUMN, args=(name,))

    @classmethod
    def const(cls, value: Union[int, float]) -> "Expr":
        """Create constant expression."""
        return cls(OpCode.CONSTANT, args=(value,))

    @classmethod
    def from_data(cls, data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> "Expr":
        """Wrap existing data as expression."""
        expr = cls(OpCode.CONSTANT)
        expr._cached_data = data
        return expr

    # --- Time-Series Operations (Rolling) ---

    def rolling(self, window: int, min_periods: int = None) -> "_RollingExpr":
        """Create rolling window context."""
        return _RollingExpr(self, window, min_periods)

    def expanding(self, min_periods: int = 1) -> "_ExpandingExpr":
        """Create expanding window context."""
        return _ExpandingExpr(self, min_periods)

    def ewm(self, span: int = None, alpha: float = None, halflife: float = None) -> "_EWMExpr":
        """Create exponentially weighted window context."""
        return _EWMExpr(self, span=span, alpha=alpha, halflife=halflife)

    def shift(self, periods: int = 1, fill_value: Any = None) -> "Expr":
        """Shift values by periods."""
        return Expr(OpCode.TS_DELAY, args=(periods,), kwargs={"fill_value": fill_value}, inputs=[self])

    def diff(self, periods: int = 1) -> "Expr":
        """Difference with previous value."""
        return Expr(OpCode.TS_DELTA, args=(periods,), inputs=[self])

    def pct_change(self, periods: int = 1, fill_method: str = None) -> "Expr":
        """Percentage change."""
        shifted = self.shift(periods)
        return (self - shifted) / shifted

    # --- Cross-Sectional Operations ---

    def rank(self, axis: int = 1, pct: bool = True, method: str = "average") -> "Expr":
        """Cross-sectional rank."""
        return Expr(OpCode.CS_RANK, kwargs={"axis": axis, "pct": pct, "method": method}, inputs=[self])

    def demean(self, axis: int = 1) -> "Expr":
        """Cross-sectional demean."""
        return Expr(OpCode.CS_DEMEAN, kwargs={"axis": axis}, inputs=[self])

    def zscore(self, axis: int = 1) -> "Expr":
        """Cross-sectional z-score."""
        return Expr(OpCode.CS_ZSCORE, kwargs={"axis": axis}, inputs=[self])

    def l1_norm(self, axis: int = 1) -> "Expr":
        """L1 normalize (sum of abs = 1)."""
        return Expr(OpCode.CS_L1_NORM, kwargs={"axis": axis}, inputs=[self])

    def l2_norm(self, axis: int = 1) -> "Expr":
        """L2 normalize (sum of squares = 1)."""
        return Expr(OpCode.CS_L2_NORM, kwargs={"axis": axis}, inputs=[self])

    def clip(self, lower: float = None, upper: float = None) -> "Expr":
        """Clip values."""
        return Expr(OpCode.CS_CLIP, kwargs={"lower": lower, "upper": upper}, inputs=[self])

    def winsorize(self, lower: float = 0.01, upper: float = 0.99) -> "Expr":
        """Winsorize extreme values."""
        return Expr(OpCode.CS_WINSORIZE, kwargs={"lower": lower, "upper": upper}, inputs=[self])

    def softmax(self, axis: int = 1, temperature: float = 1.0) -> "Expr":
        """Softmax transformation."""
        return Expr(OpCode.CS_SOFTMAX, kwargs={"axis": axis, "temperature": temperature}, inputs=[self])

    # --- Reduction Operations ---

    def sum(self, axis: int = 1, skipna: bool = True) -> "Expr":
        """Sum along axis."""
        return Expr(OpCode.SUM, kwargs={"axis": axis, "skipna": skipna}, inputs=[self])

    def mean(self, axis: int = 1, skipna: bool = True) -> "Expr":
        """Mean along axis."""
        return Expr(OpCode.MEAN, kwargs={"axis": axis, "skipna": skipna}, inputs=[self])

    def std(self, axis: int = 1, skipna: bool = True, ddof: int = 1) -> "Expr":
        """Standard deviation along axis."""
        return Expr(OpCode.STD, kwargs={"axis": axis, "skipna": skipna, "ddof": ddof}, inputs=[self])

    def var(self, axis: int = 1, skipna: bool = True, ddof: int = 1) -> "Expr":
        """Variance along axis."""
        return Expr(OpCode.VAR, kwargs={"axis": axis, "skipna": skipna, "ddof": ddof}, inputs=[self])

    def min(self, axis: int = 1, skipna: bool = True) -> "Expr":
        """Minimum along axis."""
        return Expr(OpCode.MIN, kwargs={"axis": axis, "skipna": skipna}, inputs=[self])

    def max(self, axis: int = 1, skipna: bool = True) -> "Expr":
        """Maximum along axis."""
        return Expr(OpCode.MAX, kwargs={"axis": axis, "skipna": skipna}, inputs=[self])

    def median(self, axis: int = 1, skipna: bool = True) -> "Expr":
        """Median along axis."""
        return Expr(OpCode.MEDIAN, kwargs={"axis": axis, "skipna": skipna}, inputs=[self])

    def prod(self, axis: int = 1, skipna: bool = True) -> "Expr":
        """Product along axis."""
        return Expr(OpCode.PROD, kwargs={"axis": axis, "skipna": skipna}, inputs=[self])

    def count(self, axis: int = 1) -> "Expr":
        """Count non-NA values along axis."""
        return Expr(OpCode.COUNT, kwargs={"axis": axis}, inputs=[self])

    def quantile(self, q: float, axis: int = 1) -> "Expr":
        """Quantile along axis."""
        return Expr(OpCode.QUANTILE, kwargs={"q": q, "axis": axis}, inputs=[self])

    # --- Cumulative Operations ---

    def cumsum(self, axis: int = 0, skipna: bool = True) -> "Expr":
        """Cumulative sum."""
        return Expr(OpCode.CUMSUM, kwargs={"axis": axis, "skipna": skipna}, inputs=[self])

    def cumprod(self, axis: int = 0, skipna: bool = True) -> "Expr":
        """Cumulative product."""
        return Expr(OpCode.CUMPROD, kwargs={"axis": axis, "skipna": skipna}, inputs=[self])

    def cummax(self, axis: int = 0, skipna: bool = True) -> "Expr":
        """Cumulative maximum."""
        return Expr(OpCode.CUMMAX, kwargs={"axis": axis, "skipna": skipna}, inputs=[self])

    def cummin(self, axis: int = 0, skipna: bool = True) -> "Expr":
        """Cumulative minimum."""
        return Expr(OpCode.CUMMIN, kwargs={"axis": axis, "skipna": skipna}, inputs=[self])

    # --- Element-wise Math ---

    def abs(self) -> "Expr":
        """Absolute value."""
        return Expr(OpCode.ABS, inputs=[self])

    def log(self) -> "Expr":
        """Natural logarithm."""
        return Expr(OpCode.LOG, inputs=[self])

    def log10(self) -> "Expr":
        """Base-10 logarithm."""
        return Expr(OpCode.LOG10, inputs=[self])

    def log2(self) -> "Expr":
        """Base-2 logarithm."""
        return Expr(OpCode.LOG2, inputs=[self])

    def sqrt(self) -> "Expr":
        """Square root."""
        return Expr(OpCode.SQRT, inputs=[self])

    def exp(self) -> "Expr":
        """Exponential."""
        return Expr(OpCode.EXP, inputs=[self])

    def sign(self) -> "Expr":
        """Sign function."""
        return Expr(OpCode.SIGN, inputs=[self])

    def floor(self) -> "Expr":
        """Floor."""
        return Expr(OpCode.FLOOR, inputs=[self])

    def ceil(self) -> "Expr":
        """Ceiling."""
        return Expr(OpCode.CEIL, inputs=[self])

    def round(self, decimals: int = 0) -> "Expr":
        """Round to decimals."""
        return Expr(OpCode.ROUND, kwargs={"decimals": decimals}, inputs=[self])

    # --- DataFrame/Series Operations ---

    def fillna(self, value: Any = None, method: str = None) -> "Expr":
        """Fill NA values."""
        return Expr(OpCode.FILLNA, kwargs={"value": value, "method": method}, inputs=[self])

    def ffill(self, limit: int = None) -> "Expr":
        """Forward fill NA values."""
        return Expr(OpCode.FFILL, kwargs={"limit": limit}, inputs=[self])

    def bfill(self, limit: int = None) -> "Expr":
        """Backward fill NA values."""
        return Expr(OpCode.BFILL, kwargs={"limit": limit}, inputs=[self])

    def dropna(self, axis: int = 0, how: str = "any") -> "Expr":
        """Drop NA values."""
        return Expr(OpCode.DROPNA, kwargs={"axis": axis, "how": how}, inputs=[self])

    def replace(self, to_replace: Any, value: Any) -> "Expr":
        """Replace values."""
        return Expr(OpCode.REPLACE, kwargs={"to_replace": to_replace, "value": value}, inputs=[self])

    def reindex(self, index: Any = None, columns: Any = None, method: str = None, fill_value: Any = None) -> "Expr":
        """Reindex to new index/columns."""
        return Expr(OpCode.REINDEX, kwargs={
            "index": index, "columns": columns, "method": method, "fill_value": fill_value
        }, inputs=[self])

    def interpolate(self, method: str = "linear", axis: int = 0) -> "Expr":
        """Interpolate NA values."""
        return Expr(OpCode.INTERPOLATE, kwargs={"method": method, "axis": axis}, inputs=[self])

    def apply(self, func: Callable, axis: int = 0, **kwargs) -> "Expr":
        """Apply function along axis."""
        return Expr(OpCode.APPLY, kwargs={"func": func, "axis": axis, **kwargs}, inputs=[self])

    # --- Conditional Operations ---

    def where(self, cond: "Expr", other: Any = np.nan) -> "Expr":
        """Where condition is True, keep self; otherwise use other."""
        cond_expr = _ensure_expr(cond)
        other_expr = _ensure_expr(other)
        return Expr(OpCode.CONDITION, kwargs={"keep_self": True}, inputs=[self, cond_expr, other_expr])

    def mask(self, cond: "Expr", other: Any = np.nan) -> "Expr":
        """Where condition is True, use other; otherwise keep self."""
        cond_expr = _ensure_expr(cond)
        other_expr = _ensure_expr(other)
        return Expr(OpCode.CONDITION, kwargs={"keep_self": False}, inputs=[self, cond_expr, other_expr])

    # --- Operator Overloading - Arithmetic ---

    def __add__(self, other) -> "Expr":
        other = _ensure_expr(other)
        return Expr(OpCode.ADD, inputs=[self, other])

    def __radd__(self, other) -> "Expr":
        other = _ensure_expr(other)
        return Expr(OpCode.ADD, inputs=[other, self])

    def __sub__(self, other) -> "Expr":
        other = _ensure_expr(other)
        return Expr(OpCode.SUB, inputs=[self, other])

    def __rsub__(self, other) -> "Expr":
        other = _ensure_expr(other)
        return Expr(OpCode.SUB, inputs=[other, self])

    def __mul__(self, other) -> "Expr":
        other = _ensure_expr(other)
        return Expr(OpCode.MUL, inputs=[self, other])

    def __rmul__(self, other) -> "Expr":
        other = _ensure_expr(other)
        return Expr(OpCode.MUL, inputs=[other, self])

    def __truediv__(self, other) -> "Expr":
        other = _ensure_expr(other)
        return Expr(OpCode.DIV, inputs=[self, other])

    def __rtruediv__(self, other) -> "Expr":
        other = _ensure_expr(other)
        return Expr(OpCode.DIV, inputs=[other, self])

    def __floordiv__(self, other) -> "Expr":
        other = _ensure_expr(other)
        return (self / other).floor()

    def __rfloordiv__(self, other) -> "Expr":
        other = _ensure_expr(other)
        return (other / self).floor()

    def __mod__(self, other) -> "Expr":
        other = _ensure_expr(other)
        return self - (self // other) * other

    def __rmod__(self, other) -> "Expr":
        other = _ensure_expr(other)
        return other - (other // self) * self

    def __pow__(self, other) -> "Expr":
        other = _ensure_expr(other)
        return Expr(OpCode.POW, inputs=[self, other])

    def __rpow__(self, other) -> "Expr":
        other = _ensure_expr(other)
        return Expr(OpCode.POW, inputs=[other, self])

    def __neg__(self) -> "Expr":
        return Expr(OpCode.NEG, inputs=[self])

    def __pos__(self) -> "Expr":
        return self

    def __abs__(self) -> "Expr":
        return self.abs()

    def __invert__(self) -> "Expr":
        """Bitwise NOT (~) - used for boolean inversion."""
        return Expr(OpCode.NOT, inputs=[self])

    # --- Operator Overloading - Comparison ---

    def __lt__(self, other) -> "Expr":
        other = _ensure_expr(other)
        return Expr(OpCode.LT, inputs=[self, other])

    def __le__(self, other) -> "Expr":
        other = _ensure_expr(other)
        return Expr(OpCode.LE, inputs=[self, other])

    def __gt__(self, other) -> "Expr":
        other = _ensure_expr(other)
        return Expr(OpCode.GT, inputs=[self, other])

    def __ge__(self, other) -> "Expr":
        other = _ensure_expr(other)
        return Expr(OpCode.GE, inputs=[self, other])

    def __eq__(self, other) -> "Expr":
        other = _ensure_expr(other)
        return Expr(OpCode.EQ, inputs=[self, other])

    def __ne__(self, other) -> "Expr":
        other = _ensure_expr(other)
        return Expr(OpCode.NE, inputs=[self, other])

    def __and__(self, other) -> "Expr":
        other = _ensure_expr(other)
        return Expr(OpCode.AND, inputs=[self, other])

    def __rand__(self, other) -> "Expr":
        other = _ensure_expr(other)
        return Expr(OpCode.AND, inputs=[other, self])

    def __or__(self, other) -> "Expr":
        other = _ensure_expr(other)
        return Expr(OpCode.OR, inputs=[self, other])

    def __ror__(self, other) -> "Expr":
        other = _ensure_expr(other)
        return Expr(OpCode.OR, inputs=[other, self])

    # --- Numpy ufunc support (interception) ---

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Intercept numpy ufuncs to return Expr."""
        if method != "__call__":
            return NotImplemented

        # Map numpy ufuncs to OpCodes
        ufunc_map = {
            np.add: OpCode.ADD,
            np.subtract: OpCode.SUB,
            np.multiply: OpCode.MUL,
            np.divide: OpCode.DIV,
            np.true_divide: OpCode.DIV,
            np.power: OpCode.POW,
            np.negative: OpCode.NEG,
            np.positive: None,  # No-op
            np.absolute: OpCode.ABS,
            np.abs: OpCode.ABS,
            np.sqrt: OpCode.SQRT,
            np.log: OpCode.LOG,
            np.log10: OpCode.LOG10,
            np.log2: OpCode.LOG2,
            np.exp: OpCode.EXP,
            np.sign: OpCode.SIGN,
            np.floor: OpCode.FLOOR,
            np.ceil: OpCode.CEIL,
            np.minimum: OpCode.ELEM_MIN,
            np.maximum: OpCode.ELEM_MAX,
            np.less: OpCode.LT,
            np.less_equal: OpCode.LE,
            np.greater: OpCode.GT,
            np.greater_equal: OpCode.GE,
            np.equal: OpCode.EQ,
            np.not_equal: OpCode.NE,
            np.logical_and: OpCode.AND,
            np.logical_or: OpCode.OR,
            np.logical_not: OpCode.NOT,
        }

        op_code = ufunc_map.get(ufunc)
        if op_code is None:
            if ufunc == np.positive:
                # Just return self for positive
                for inp in inputs:
                    if isinstance(inp, Expr):
                        return inp
            # Unknown ufunc - wrap as generic numpy func
            expr_inputs = [_ensure_expr(inp) for inp in inputs]
            return Expr(OpCode.NUMPY_FUNC, kwargs={"ufunc": ufunc.__name__, **kwargs}, inputs=expr_inputs)

        # Convert inputs to Expr
        expr_inputs = [_ensure_expr(inp) for inp in inputs]

        # Unary operations
        if len(expr_inputs) == 1:
            return Expr(op_code, inputs=expr_inputs)

        # Binary operations
        return Expr(op_code, inputs=expr_inputs)

    # --- Execution ---

    def execute(self, data: Optional[pd.DataFrame] = None) -> Union[pd.DataFrame, pd.Series]:
        """Execute expression tree with pandas."""
        return _execute_pandas(self, data)

    def to_spec(self) -> Dict[str, Any]:
        """Serialize expression tree to specification for Rust backend."""
        spec = {
            "op": self.op.value,
            "args": list(self.args),
            "kwargs": {k: v for k, v in self.kwargs.items() if not callable(v)},
            "inputs": [inp.to_spec() for inp in self.inputs],
        }
        return spec

    def __repr__(self) -> str:
        if self.op == OpCode.COLUMN:
            return f"col({self.args[0]!r})"
        elif self.op == OpCode.CONSTANT:
            if hasattr(self, "_cached_data"):
                return f"data({type(self._cached_data).__name__})"
            return f"const({self.args[0]})"
        else:
            inputs_str = ", ".join(repr(inp) for inp in self.inputs)
            return f"{self.op.value}({inputs_str})"


# --- Rolling/Expanding/EWM Window Contexts ---

class _RollingExpr:
    """Rolling window context for Expr."""

    def __init__(self, expr: Expr, window: int, min_periods: int = None):
        self.expr = expr
        self.window = window
        self.min_periods = min_periods or window

    def mean(self) -> Expr:
        return Expr(OpCode.TS_MEAN, args=(self.window,), kwargs={"min_periods": self.min_periods}, inputs=[self.expr])

    def std(self, ddof: int = 1) -> Expr:
        return Expr(OpCode.TS_STD, args=(self.window,), kwargs={"min_periods": self.min_periods, "ddof": ddof}, inputs=[self.expr])

    def var(self, ddof: int = 1) -> Expr:
        return Expr(OpCode.VAR, args=(self.window,), kwargs={"min_periods": self.min_periods, "ddof": ddof}, inputs=[self.expr])

    def sum(self) -> Expr:
        return Expr(OpCode.TS_SUM, args=(self.window,), kwargs={"min_periods": self.min_periods}, inputs=[self.expr])

    def min(self) -> Expr:
        return Expr(OpCode.TS_MIN, args=(self.window,), kwargs={"min_periods": self.min_periods}, inputs=[self.expr])

    def max(self) -> Expr:
        return Expr(OpCode.TS_MAX, args=(self.window,), kwargs={"min_periods": self.min_periods}, inputs=[self.expr])

    def rank(self, pct: bool = True) -> Expr:
        return Expr(OpCode.TS_RANK, args=(self.window,), kwargs={"min_periods": self.min_periods, "pct": pct}, inputs=[self.expr])

    def median(self) -> Expr:
        return Expr(OpCode.MEDIAN, args=(self.window,), kwargs={"min_periods": self.min_periods}, inputs=[self.expr])

    def quantile(self, q: float) -> Expr:
        return Expr(OpCode.TS_QUANTILE, args=(self.window,), kwargs={"q": q, "min_periods": self.min_periods}, inputs=[self.expr])

    def count(self) -> Expr:
        return Expr(OpCode.COUNT, args=(self.window,), kwargs={"min_periods": self.min_periods}, inputs=[self.expr])

    def apply(self, func: Callable, raw: bool = True) -> Expr:
        return Expr(OpCode.APPLY, args=(self.window,), kwargs={"func": func, "raw": raw, "min_periods": self.min_periods}, inputs=[self.expr])

    def corr(self, other: Expr) -> Expr:
        other_expr = _ensure_expr(other)
        return Expr(OpCode.TS_CORR, args=(self.window,), kwargs={"min_periods": self.min_periods}, inputs=[self.expr, other_expr])

    def cov(self, other: Expr) -> Expr:
        other_expr = _ensure_expr(other)
        return Expr(OpCode.TS_COV, args=(self.window,), kwargs={"min_periods": self.min_periods}, inputs=[self.expr, other_expr])


class _ExpandingExpr:
    """Expanding window context for Expr."""

    def __init__(self, expr: Expr, min_periods: int = 1):
        self.expr = expr
        self.min_periods = min_periods

    def mean(self) -> Expr:
        return Expr(OpCode.MEAN, kwargs={"expanding": True, "min_periods": self.min_periods}, inputs=[self.expr])

    def std(self, ddof: int = 1) -> Expr:
        return Expr(OpCode.STD, kwargs={"expanding": True, "min_periods": self.min_periods, "ddof": ddof}, inputs=[self.expr])

    def sum(self) -> Expr:
        return Expr(OpCode.SUM, kwargs={"expanding": True, "min_periods": self.min_periods}, inputs=[self.expr])

    def min(self) -> Expr:
        return Expr(OpCode.MIN, kwargs={"expanding": True, "min_periods": self.min_periods}, inputs=[self.expr])

    def max(self) -> Expr:
        return Expr(OpCode.MAX, kwargs={"expanding": True, "min_periods": self.min_periods}, inputs=[self.expr])


class _EWMExpr:
    """Exponentially weighted window context for Expr."""

    def __init__(self, expr: Expr, span: int = None, alpha: float = None, halflife: float = None):
        self.expr = expr
        self.span = span
        self.alpha = alpha
        self.halflife = halflife

    def mean(self) -> Expr:
        return Expr(OpCode.EMA, kwargs={"span": self.span, "alpha": self.alpha, "halflife": self.halflife}, inputs=[self.expr])

    def std(self) -> Expr:
        return Expr(OpCode.STD, kwargs={"ewm": True, "span": self.span, "alpha": self.alpha}, inputs=[self.expr])

    def var(self) -> Expr:
        return Expr(OpCode.VAR, kwargs={"ewm": True, "span": self.span, "alpha": self.alpha}, inputs=[self.expr])


# --- Helper Functions ---

def _ensure_expr(value: Any) -> Expr:
    """Convert value to Expr if needed."""
    if isinstance(value, Expr):
        return value
    elif isinstance(value, (int, float, np.integer, np.floating)):
        return Expr.const(float(value))
    elif isinstance(value, (pd.DataFrame, pd.Series, np.ndarray)):
        return Expr.from_data(value)
    elif value is None or (isinstance(value, float) and np.isnan(value)):
        return Expr.const(np.nan)
    else:
        # Try to convert to float
        try:
            return Expr.const(float(value))
        except (TypeError, ValueError):
            raise TypeError(f"Cannot convert {type(value)} to Expr")


# --- Pandas Execution Engine ---

def _execute_pandas(expr: Expr, data: Optional[pd.DataFrame]) -> Union[pd.DataFrame, pd.Series, np.ndarray, float]:
    """Execute expression tree with pandas."""

    # Leaf nodes
    if expr.op == OpCode.COLUMN:
        col_name = expr.args[0]
        if data is None:
            raise ValueError(f"Data required for column reference: {col_name}")
        return data[col_name] if col_name in data.columns else data

    if expr.op == OpCode.CONSTANT:
        if hasattr(expr, "_cached_data"):
            return expr._cached_data
        return expr.args[0] if expr.args else np.nan

    # Recursive evaluation of inputs
    inputs = [_execute_pandas(inp, data) for inp in expr.inputs]

    # --- Time-series operations ---
    if expr.op == OpCode.TS_MEAN:
        window = expr.args[0]
        min_periods = expr.kwargs.get("min_periods", window)
        return inputs[0].rolling(window, min_periods=min_periods).mean()

    elif expr.op == OpCode.TS_STD:
        window = expr.args[0]
        min_periods = expr.kwargs.get("min_periods", window)
        ddof = expr.kwargs.get("ddof", 1)
        return inputs[0].rolling(window, min_periods=min_periods).std(ddof=ddof)

    elif expr.op == OpCode.TS_SUM:
        window = expr.args[0]
        min_periods = expr.kwargs.get("min_periods", window)
        return inputs[0].rolling(window, min_periods=min_periods).sum()

    elif expr.op == OpCode.TS_MIN:
        window = expr.args[0]
        min_periods = expr.kwargs.get("min_periods", window)
        return inputs[0].rolling(window, min_periods=min_periods).min()

    elif expr.op == OpCode.TS_MAX:
        window = expr.args[0]
        min_periods = expr.kwargs.get("min_periods", window)
        return inputs[0].rolling(window, min_periods=min_periods).max()

    elif expr.op == OpCode.TS_RANK:
        window = expr.args[0]
        min_periods = expr.kwargs.get("min_periods", window)
        pct = expr.kwargs.get("pct", True)
        return inputs[0].rolling(window, min_periods=min_periods).rank(pct=pct)

    elif expr.op == OpCode.TS_DELTA:
        periods = expr.args[0]
        return inputs[0].diff(periods)

    elif expr.op == OpCode.TS_DELAY:
        periods = expr.args[0]
        fill_value = expr.kwargs.get("fill_value")
        return inputs[0].shift(periods, fill_value=fill_value)

    elif expr.op == OpCode.TS_CORR:
        window = expr.args[0]
        min_periods = expr.kwargs.get("min_periods", window)
        return inputs[0].rolling(window, min_periods=min_periods).corr(inputs[1])

    elif expr.op == OpCode.TS_COV:
        window = expr.args[0]
        min_periods = expr.kwargs.get("min_periods", window)
        return inputs[0].rolling(window, min_periods=min_periods).cov(inputs[1])

    elif expr.op == OpCode.TS_QUANTILE:
        window = expr.args[0]
        q = expr.kwargs.get("q", 0.5)
        min_periods = expr.kwargs.get("min_periods", window)
        return inputs[0].rolling(window, min_periods=min_periods).quantile(q)

    elif expr.op == OpCode.TS_ZSCORE:
        window = expr.args[0]
        mean = inputs[0].rolling(window).mean()
        std = inputs[0].rolling(window).std()
        return (inputs[0] - mean) / std.replace(0, np.nan)

    elif expr.op == OpCode.TS_SCALE:
        window = expr.args[0]
        constant = expr.kwargs.get("constant", 0)
        min_val = inputs[0].rolling(window).min()
        max_val = inputs[0].rolling(window).max()
        range_val = max_val - min_val
        return (inputs[0] - min_val) / range_val.replace(0, np.nan) + constant

    elif expr.op == OpCode.TS_PRODUCT:
        window = expr.args[0]
        return inputs[0].rolling(window).apply(np.prod, raw=True)

    elif expr.op == OpCode.TS_DECAYED_LINEAR:
        window = expr.args[0]
        weights = np.arange(1, window + 1, dtype=float)
        weights = weights / weights.sum()
        return inputs[0].rolling(window).apply(lambda x: np.dot(x, weights), raw=True)

    elif expr.op == OpCode.TS_ARGMAX:
        window = expr.args[0]
        return inputs[0].rolling(window).apply(lambda x: np.argmax(x), raw=True)

    elif expr.op == OpCode.TS_ARGMIN:
        window = expr.args[0]
        return inputs[0].rolling(window).apply(lambda x: np.argmin(x), raw=True)

    elif expr.op == OpCode.TS_SLOPE:
        window = expr.args[0]
        def calc_slope(y):
            if len(y) < 2:
                return np.nan
            x = np.arange(len(y))
            return np.polyfit(x, y, 1)[0]
        return inputs[0].rolling(window).apply(calc_slope, raw=True)

    elif expr.op == OpCode.LOWDAY:
        window = expr.args[0]
        return inputs[0].rolling(window).apply(lambda x: len(x) - np.argmin(x) - 1, raw=True)

    elif expr.op == OpCode.HIGHDAY:
        window = expr.args[0]
        return inputs[0].rolling(window).apply(lambda x: len(x) - np.argmax(x) - 1, raw=True)

    elif expr.op == OpCode.TS_RETURNS:
        # pct_change
        periods = expr.args[0] if expr.args else 1
        return inputs[0].pct_change(periods=periods)

    elif expr.op == OpCode.TS_LOG_RETURNS:
        # log returns: log(p_t / p_{t-1})
        periods = expr.args[0] if expr.args else 1
        return np.log(inputs[0] / inputs[0].shift(periods))

    elif expr.op == OpCode.TS_CUMSUM:
        # cumulative sum
        axis = expr.kwargs.get("axis", 0)
        return inputs[0].cumsum(axis=axis)

    elif expr.op == OpCode.TS_CUMPROD:
        # cumulative product
        axis = expr.kwargs.get("axis", 0)
        return inputs[0].cumprod(axis=axis)

    # --- Cross-sectional operations ---
    elif expr.op == OpCode.CS_RANK:
        axis = expr.kwargs.get("axis", 1)
        pct = expr.kwargs.get("pct", True)
        method = expr.kwargs.get("method", "average")
        return inputs[0].rank(axis=axis, pct=pct, method=method)

    elif expr.op == OpCode.CS_DEMEAN:
        axis = expr.kwargs.get("axis", 1)
        return inputs[0].sub(inputs[0].mean(axis=axis), axis=0)

    elif expr.op == OpCode.CS_ZSCORE:
        axis = expr.kwargs.get("axis", 1)
        mean = inputs[0].mean(axis=axis)
        std = inputs[0].std(axis=axis)
        return inputs[0].sub(mean, axis=0).div(std.replace(0, np.nan), axis=0)

    elif expr.op == OpCode.CS_L1_NORM:
        axis = expr.kwargs.get("axis", 1)
        abs_sum = inputs[0].abs().sum(axis=axis)
        return inputs[0].div(abs_sum.replace(0, np.nan), axis=0)

    elif expr.op == OpCode.CS_L2_NORM:
        axis = expr.kwargs.get("axis", 1)
        l2 = np.sqrt((inputs[0] ** 2).sum(axis=axis))
        return inputs[0].div(l2.replace(0, np.nan), axis=0)

    elif expr.op == OpCode.CS_CLIP:
        lower = expr.kwargs.get("lower")
        upper = expr.kwargs.get("upper")
        return inputs[0].clip(lower=lower, upper=upper)

    elif expr.op == OpCode.CS_WINSORIZE:
        lower = expr.kwargs.get("lower", 0.01)
        upper = expr.kwargs.get("upper", 0.99)
        q_low = inputs[0].quantile(lower, axis=1)
        q_high = inputs[0].quantile(upper, axis=1)
        return inputs[0].clip(lower=q_low, upper=q_high, axis=0)

    elif expr.op == OpCode.CS_SOFTMAX:
        axis = expr.kwargs.get("axis", 1)
        temp = expr.kwargs.get("temperature", 1.0)
        shifted = inputs[0].sub(inputs[0].max(axis=axis), axis=0) / temp
        exp_data = np.exp(shifted)
        return exp_data.div(exp_data.sum(axis=axis), axis=0)

    elif expr.op == OpCode.CS_SCALE:
        axis = expr.kwargs.get("axis", 1)
        scale_val = expr.kwargs.get("scale_val", 1.0)
        abs_sum = inputs[0].abs().sum(axis=axis)
        return inputs[0].div(abs_sum.replace(0, np.nan), axis=0) * scale_val

    elif expr.op == OpCode.CS_QUANTILE:
        distribution = expr.kwargs.get("distribution", "uniform")
        if distribution == "gaussian":
            from scipy import stats
            ranked = inputs[0].rank(axis=1, pct=True).clip(0.001, 0.999)
            return ranked.apply(lambda x: stats.norm.ppf(x))
        return inputs[0].rank(axis=1, pct=True)

    elif expr.op == OpCode.CS_TOPK_MASK:
        k = expr.kwargs.get("k", 10)
        ascending = expr.kwargs.get("ascending", False)
        ranks = inputs[0].rank(axis=1, ascending=ascending)
        return ranks <= k

    elif expr.op == OpCode.CS_GROUPED_DEMEAN:
        groups = expr.kwargs.get("groups")  # Dict[str, str]: symbol -> group
        data = inputs[0]
        if groups is None:
            # No groups - just demean across all
            return data.sub(data.mean(axis=1), axis=0)
        # Group-based demeaning
        if isinstance(data, pd.DataFrame):
            result = data.copy()
            for timestamp in data.index:
                row = data.loc[timestamp]
                # Group by sector
                sector_groups = {}
                for symbol in row.index:
                    if pd.notna(row[symbol]):
                        sector = groups.get(symbol, "Unknown")
                        if sector not in sector_groups:
                            sector_groups[sector] = []
                        sector_groups[sector].append(symbol)
                # Demean within each sector
                for sector, symbols in sector_groups.items():
                    sector_values = row[symbols]
                    sector_mean = sector_values.mean()
                    result.loc[timestamp, symbols] = sector_values - sector_mean
            return result
        else:
            # Series case
            sector_groups = {}
            for symbol, value in data.items():
                if pd.notna(value):
                    sector = groups.get(symbol, "Unknown")
                    if sector not in sector_groups:
                        sector_groups[sector] = []
                    sector_groups[sector].append(symbol)
            result = data.copy()
            for sector, symbols in sector_groups.items():
                sector_values = data[symbols]
                sector_mean = sector_values.mean()
                result[symbols] = sector_values - sector_mean
            return result

    # --- Strata-based operations (by_* operators) ---
    elif expr.op == OpCode.BY_DEMEAN:
        strata = expr.kwargs.get("strata")  # Dict[str, str]: symbol -> category
        data = inputs[0]
        if strata is None:
            # No strata - just demean across all
            return data.sub(data.mean(axis=1), axis=0)
        return _by_operation(data, strata, "demean")

    elif expr.op == OpCode.BY_RANK:
        strata = expr.kwargs.get("strata")
        pct = expr.kwargs.get("pct", True)
        data = inputs[0]
        if strata is None:
            return data.rank(axis=1, pct=pct)
        return _by_operation(data, strata, "rank", pct=pct)

    elif expr.op == OpCode.BY_ZSCORE:
        strata = expr.kwargs.get("strata")
        data = inputs[0]
        if strata is None:
            mean = data.mean(axis=1)
            std = data.std(axis=1)
            return data.sub(mean, axis=0).div(std.replace(0, np.nan), axis=0)
        return _by_operation(data, strata, "zscore")

    elif expr.op == OpCode.BY_SCALE:
        strata = expr.kwargs.get("strata")
        scale_val = expr.kwargs.get("scale_val", 1.0)
        data = inputs[0]
        if strata is None:
            abs_sum = data.abs().sum(axis=1)
            return data.div(abs_sum.replace(0, np.nan), axis=0) * scale_val
        return _by_operation(data, strata, "scale", scale_val=scale_val)

    elif expr.op == OpCode.BY_MEAN:
        strata = expr.kwargs.get("strata")
        data = inputs[0]
        if strata is None:
            return data.mean(axis=1)
        return _by_operation(data, strata, "mean")

    elif expr.op == OpCode.BY_SUM:
        strata = expr.kwargs.get("strata")
        data = inputs[0]
        if strata is None:
            return data.sum(axis=1)
        return _by_operation(data, strata, "sum")

    elif expr.op == OpCode.BY_MAX:
        strata = expr.kwargs.get("strata")
        data = inputs[0]
        if strata is None:
            return data.max(axis=1)
        return _by_operation(data, strata, "max")

    elif expr.op == OpCode.BY_MIN:
        strata = expr.kwargs.get("strata")
        data = inputs[0]
        if strata is None:
            return data.min(axis=1)
        return _by_operation(data, strata, "min")

    elif expr.op == OpCode.BY_STD:
        strata = expr.kwargs.get("strata")
        data = inputs[0]
        if strata is None:
            return data.std(axis=1)
        return _by_operation(data, strata, "std")

    elif expr.op == OpCode.BY_MEDIAN:
        strata = expr.kwargs.get("strata")
        data = inputs[0]
        if strata is None:
            return data.median(axis=1)
        return _by_operation(data, strata, "median")

    elif expr.op == OpCode.BY_COUNT:
        strata = expr.kwargs.get("strata")
        data = inputs[0]
        if strata is None:
            return data.count(axis=1)
        return _by_operation(data, strata, "count")

    elif expr.op == OpCode.BINNING:
        n_bins = expr.kwargs.get("n_bins", 5)
        labels = expr.kwargs.get("labels")
        data = inputs[0]
        return _binning(data, n_bins=n_bins, labels=labels)

    # --- Reduction operations ---
    elif expr.op == OpCode.SUM:
        axis = expr.kwargs.get("axis", 1)
        skipna = expr.kwargs.get("skipna", True)
        if expr.kwargs.get("expanding"):
            return inputs[0].expanding(min_periods=expr.kwargs.get("min_periods", 1)).sum()
        return inputs[0].sum(axis=axis, skipna=skipna)

    elif expr.op == OpCode.MEAN:
        axis = expr.kwargs.get("axis", 1)
        skipna = expr.kwargs.get("skipna", True)
        if expr.kwargs.get("expanding"):
            return inputs[0].expanding(min_periods=expr.kwargs.get("min_periods", 1)).mean()
        return inputs[0].mean(axis=axis, skipna=skipna)

    elif expr.op == OpCode.STD:
        axis = expr.kwargs.get("axis", 1)
        skipna = expr.kwargs.get("skipna", True)
        ddof = expr.kwargs.get("ddof", 1)
        if expr.kwargs.get("expanding"):
            return inputs[0].expanding(min_periods=expr.kwargs.get("min_periods", 1)).std(ddof=ddof)
        if expr.kwargs.get("ewm"):
            return inputs[0].ewm(span=expr.kwargs.get("span")).std()
        return inputs[0].std(axis=axis, skipna=skipna, ddof=ddof)

    elif expr.op == OpCode.VAR:
        axis = expr.kwargs.get("axis", 1)
        skipna = expr.kwargs.get("skipna", True)
        ddof = expr.kwargs.get("ddof", 1)
        if expr.kwargs.get("ewm"):
            return inputs[0].ewm(span=expr.kwargs.get("span")).var()
        return inputs[0].var(axis=axis, skipna=skipna, ddof=ddof)

    elif expr.op == OpCode.MIN:
        axis = expr.kwargs.get("axis", 1)
        skipna = expr.kwargs.get("skipna", True)
        if expr.kwargs.get("expanding"):
            return inputs[0].expanding(min_periods=expr.kwargs.get("min_periods", 1)).min()
        return inputs[0].min(axis=axis, skipna=skipna)

    elif expr.op == OpCode.MAX:
        axis = expr.kwargs.get("axis", 1)
        skipna = expr.kwargs.get("skipna", True)
        if expr.kwargs.get("expanding"):
            return inputs[0].expanding(min_periods=expr.kwargs.get("min_periods", 1)).max()
        return inputs[0].max(axis=axis, skipna=skipna)

    elif expr.op == OpCode.MEDIAN:
        axis = expr.kwargs.get("axis", 1)
        if expr.args:  # Rolling median
            window = expr.args[0]
            return inputs[0].rolling(window).median()
        return inputs[0].median(axis=axis)

    elif expr.op == OpCode.PROD:
        axis = expr.kwargs.get("axis", 1)
        skipna = expr.kwargs.get("skipna", True)
        return inputs[0].prod(axis=axis, skipna=skipna)

    elif expr.op == OpCode.COUNT:
        axis = expr.kwargs.get("axis", 1)
        if expr.args:  # Rolling count
            window = expr.args[0]
            return inputs[0].rolling(window).count()
        return inputs[0].count(axis=axis)

    elif expr.op == OpCode.QUANTILE:
        q = expr.kwargs.get("q", 0.5)
        axis = expr.kwargs.get("axis", 1)
        return inputs[0].quantile(q, axis=axis)

    # --- Cumulative operations ---
    elif expr.op == OpCode.CUMSUM:
        axis = expr.kwargs.get("axis", 0)
        skipna = expr.kwargs.get("skipna", True)
        return inputs[0].cumsum(axis=axis, skipna=skipna)

    elif expr.op == OpCode.CUMPROD:
        axis = expr.kwargs.get("axis", 0)
        skipna = expr.kwargs.get("skipna", True)
        return inputs[0].cumprod(axis=axis, skipna=skipna)

    elif expr.op == OpCode.CUMMAX:
        axis = expr.kwargs.get("axis", 0)
        skipna = expr.kwargs.get("skipna", True)
        return inputs[0].cummax(axis=axis, skipna=skipna)

    elif expr.op == OpCode.CUMMIN:
        axis = expr.kwargs.get("axis", 0)
        skipna = expr.kwargs.get("skipna", True)
        return inputs[0].cummin(axis=axis, skipna=skipna)

    # --- Arithmetic (binary) ---
    elif expr.op == OpCode.ADD:
        return inputs[0] + inputs[1]

    elif expr.op == OpCode.SUB:
        return inputs[0] - inputs[1]

    elif expr.op == OpCode.MUL:
        return inputs[0] * inputs[1]

    elif expr.op == OpCode.DIV:
        return inputs[0] / inputs[1]

    elif expr.op == OpCode.POW:
        return inputs[0] ** inputs[1]

    elif expr.op == OpCode.ELEM_MIN:
        return np.minimum(inputs[0], inputs[1])

    elif expr.op == OpCode.ELEM_MAX:
        return np.maximum(inputs[0], inputs[1])

    # --- Arithmetic (unary) ---
    elif expr.op == OpCode.NEG:
        return -inputs[0]

    elif expr.op == OpCode.ABS:
        return np.abs(inputs[0])

    elif expr.op == OpCode.LOG:
        return np.log(inputs[0])

    elif expr.op == OpCode.LOG10:
        return np.log10(inputs[0])

    elif expr.op == OpCode.LOG2:
        return np.log2(inputs[0])

    elif expr.op == OpCode.SQRT:
        return np.sqrt(inputs[0])

    elif expr.op == OpCode.EXP:
        return np.exp(inputs[0])

    elif expr.op == OpCode.SIGN:
        return np.sign(inputs[0])

    elif expr.op == OpCode.FLOOR:
        return np.floor(inputs[0])

    elif expr.op == OpCode.CEIL:
        return np.ceil(inputs[0])

    elif expr.op == OpCode.ROUND:
        decimals = expr.kwargs.get("decimals", 0)
        return np.round(inputs[0], decimals=decimals)

    # --- Comparison ---
    elif expr.op == OpCode.LT:
        return inputs[0] < inputs[1]

    elif expr.op == OpCode.LE:
        return inputs[0] <= inputs[1]

    elif expr.op == OpCode.GT:
        return inputs[0] > inputs[1]

    elif expr.op == OpCode.GE:
        return inputs[0] >= inputs[1]

    elif expr.op == OpCode.EQ:
        return inputs[0] == inputs[1]

    elif expr.op == OpCode.NE:
        return inputs[0] != inputs[1]

    elif expr.op == OpCode.AND:
        return inputs[0] & inputs[1]

    elif expr.op == OpCode.OR:
        return inputs[0] | inputs[1]

    elif expr.op == OpCode.NOT:
        return ~inputs[0]

    elif expr.op == OpCode.CONDITION:
        # where/mask
        keep_self = expr.kwargs.get("keep_self", True)
        if keep_self:
            return inputs[0].where(inputs[1], inputs[2])
        else:
            return inputs[0].mask(inputs[1], inputs[2])

    # --- DataFrame/Series operations ---
    elif expr.op == OpCode.FILLNA:
        value = expr.kwargs.get("value")
        method = expr.kwargs.get("method")
        if method:
            return inputs[0].fillna(method=method)
        return inputs[0].fillna(value)

    elif expr.op == OpCode.FFILL:
        limit = expr.kwargs.get("limit")
        return inputs[0].ffill(limit=limit)

    elif expr.op == OpCode.BFILL:
        limit = expr.kwargs.get("limit")
        return inputs[0].bfill(limit=limit)

    elif expr.op == OpCode.DROPNA:
        axis = expr.kwargs.get("axis", 0)
        how = expr.kwargs.get("how", "any")
        return inputs[0].dropna(axis=axis, how=how)

    elif expr.op == OpCode.REPLACE:
        to_replace = expr.kwargs.get("to_replace")
        value = expr.kwargs.get("value")
        return inputs[0].replace(to_replace, value)

    elif expr.op == OpCode.REINDEX:
        index = expr.kwargs.get("index")
        columns = expr.kwargs.get("columns")
        method = expr.kwargs.get("method")
        fill_value = expr.kwargs.get("fill_value")
        return inputs[0].reindex(index=index, columns=columns, method=method, fill_value=fill_value)

    elif expr.op == OpCode.INTERPOLATE:
        method = expr.kwargs.get("method", "linear")
        axis = expr.kwargs.get("axis", 0)
        return inputs[0].interpolate(method=method, axis=axis)

    elif expr.op == OpCode.APPLY:
        func = expr.kwargs.get("func")
        axis = expr.kwargs.get("axis", 0)
        raw = expr.kwargs.get("raw", True)
        if expr.args:  # Rolling apply
            window = expr.args[0]
            return inputs[0].rolling(window).apply(func, raw=raw)
        return inputs[0].apply(func, axis=axis, raw=raw)

    # --- EMA ---
    elif expr.op == OpCode.EMA:
        span = expr.kwargs.get("span")
        alpha = expr.kwargs.get("alpha")
        halflife = expr.kwargs.get("halflife")
        if span:
            return inputs[0].ewm(span=span).mean()
        elif alpha:
            return inputs[0].ewm(alpha=alpha).mean()
        elif halflife:
            return inputs[0].ewm(halflife=halflife).mean()
        return inputs[0]

    # --- Generic numpy func ---
    elif expr.op == OpCode.NUMPY_FUNC:
        ufunc_name = expr.kwargs.get("ufunc")
        ufunc = getattr(np, ufunc_name)
        return ufunc(*inputs)

    # --- Pandas method fallback (for complex ops) ---
    elif expr.op == OpCode.PANDAS_METHOD:
        method = expr.kwargs.get("method")
        # Handle specific methods
        if method == "scale_down":
            constant = expr.kwargs.get("constant", 0)
            min_val = inputs[0].min(axis=1)
            max_val = inputs[0].max(axis=1)
            range_val = (max_val - min_val).replace(0, np.nan)
            return inputs[0].sub(min_val, axis=0).div(range_val, axis=0) - constant
        elif method == "ts_regbeta":
            x = np.array(expr.kwargs.get("x"))
            window = len(x)
            return inputs[0].rolling(window).apply(lambda y: np.polyfit(x, y, deg=1)[0], raw=True)
        elif method == "ts_residual":
            window = expr.kwargs.get("window")
            def calc_residual(y):
                if len(y) < 2:
                    return np.nan
                x_vals = np.arange(len(y))
                coeffs = np.polyfit(x_vals, y, 1)
                fitted = np.polyval(coeffs, x_vals)
                return y[-1] - fitted[-1]
            return inputs[0].rolling(window).apply(calc_residual, raw=True)
        elif method == "ts_rsquare":
            window = expr.kwargs.get("window")
            def calc_rsquare(vals):
                if len(vals) < 2:
                    return np.nan
                x = np.arange(len(vals))
                if np.std(vals) == 0:
                    return 0
                corr = np.corrcoef(x, vals)[0, 1]
                return corr ** 2 if not np.isnan(corr) else np.nan
            return inputs[0].rolling(window).apply(calc_rsquare, raw=True)
        elif method == "ts_linear_reg":
            window = expr.kwargs.get("window")
            mode = expr.kwargs.get("mode", 0)
            def calc_reg(y):
                if len(y) < 2:
                    return np.nan
                x = np.arange(len(y))
                coeffs = np.polyfit(x, y, 1)
                if mode == 0:
                    return coeffs[0] * (len(y) - 1) + coeffs[1]
                return coeffs[0]
            return inputs[0].rolling(window).apply(calc_reg, raw=True)
        else:
            raise NotImplementedError(f"Pandas method not implemented: {method}")

    # --- Linear Algebra - Basic Matrix Operations ---
    elif expr.op == OpCode.MATMUL:
        # matrix multiplication: A @ B
        A, B = inputs[0], inputs[1]
        if isinstance(A, (pd.DataFrame, pd.Series)):
            A = A.values
        if isinstance(B, (pd.DataFrame, pd.Series)):
            B = B.values
        return A @ B

    elif expr.op == OpCode.TRANSPOSE:
        # transpose: A.T
        A = inputs[0]
        if isinstance(A, pd.DataFrame):
            return A.T
        elif isinstance(A, np.ndarray):
            return A.T
        return np.transpose(A)

    elif expr.op == OpCode.EYE:
        # identity matrix creation
        n = expr.args[0]
        m = expr.kwargs.get("m", n)
        dtype = expr.kwargs.get("dtype", float)
        return np.eye(n, m, dtype=dtype)

    elif expr.op == OpCode.DIAG:
        # diagonal matrix creation or diagonal extraction
        A = inputs[0]
        k = expr.kwargs.get("k", 0)
        if isinstance(A, (pd.DataFrame, pd.Series)):
            A = A.values
        if A.ndim == 1:
            # vector -> diagonal matrix
            return np.diag(A, k)
        else:
            # matrix -> extract diagonal
            return np.diag(A, k)

    elif expr.op == OpCode.TRACE:
        # trace (sum of diagonal)
        A = inputs[0]
        if isinstance(A, pd.DataFrame):
            A = A.values
        return np.trace(A)

    # --- Linear Algebra - Decompositions ---
    elif expr.op == OpCode.LINALG_LU:
        # LU decomposition: A = P @ L @ U
        from scipy.linalg import lu
        A = inputs[0]
        if isinstance(A, pd.DataFrame):
            A = A.values
        P, L, U = lu(A)
        return {"P": P, "L": L, "U": U}

    elif expr.op == OpCode.LINALG_QR:
        # QR decomposition: A = Q @ R
        A = inputs[0]
        if isinstance(A, pd.DataFrame):
            A = A.values
        mode = expr.kwargs.get("mode", "reduced")
        Q, R = np.linalg.qr(A, mode=mode)
        return {"Q": Q, "R": R}

    elif expr.op == OpCode.LINALG_SVD:
        # SVD decomposition: A = U @ S @ Vh
        A = inputs[0]
        if isinstance(A, pd.DataFrame):
            A = A.values
        full_matrices = expr.kwargs.get("full_matrices", True)
        compute_uv = expr.kwargs.get("compute_uv", True)
        if compute_uv:
            U, s, Vh = np.linalg.svd(A, full_matrices=full_matrices)
            return {"U": U, "s": s, "Vh": Vh}
        else:
            s = np.linalg.svd(A, compute_uv=False)
            return {"s": s}

    elif expr.op == OpCode.LINALG_CHOLESKY:
        # Cholesky decomposition: A = L @ L.T (A is positive-definite symmetric)
        A = inputs[0]
        if isinstance(A, pd.DataFrame):
            A = A.values
        lower = expr.kwargs.get("lower", True)
        L = np.linalg.cholesky(A)
        if not lower:
            L = L.T
        return L

    elif expr.op == OpCode.LINALG_EIGEN:
        # eigenvalue decomposition
        A = inputs[0]
        if isinstance(A, pd.DataFrame):
            A = A.values
        hermitian = expr.kwargs.get("hermitian", False)
        if hermitian:
            eigenvalues, eigenvectors = np.linalg.eigh(A)
        else:
            eigenvalues, eigenvectors = np.linalg.eig(A)
        return {"eigenvalues": eigenvalues, "eigenvectors": eigenvectors}

    # --- Linear Algebra - Matrix Properties ---
    elif expr.op == OpCode.LINALG_DET:
        # determinant
        A = inputs[0]
        if isinstance(A, pd.DataFrame):
            A = A.values
        return np.linalg.det(A)

    elif expr.op == OpCode.LINALG_RANK:
        # matrix rank
        A = inputs[0]
        if isinstance(A, pd.DataFrame):
            A = A.values
        tol = expr.kwargs.get("tol", None)
        return np.linalg.matrix_rank(A, tol=tol)

    elif expr.op == OpCode.LINALG_NORM:
        # matrix/vector norm
        A = inputs[0]
        if isinstance(A, (pd.DataFrame, pd.Series)):
            A = A.values
        ord_ = expr.kwargs.get("ord", None)
        axis = expr.kwargs.get("axis", None)
        return np.linalg.norm(A, ord=ord_, axis=axis)

    elif expr.op == OpCode.LINALG_COND:
        # condition number
        A = inputs[0]
        if isinstance(A, pd.DataFrame):
            A = A.values
        p = expr.kwargs.get("p", None)
        return np.linalg.cond(A, p=p)

    # --- Linear Algebra - Solvers ---
    elif expr.op == OpCode.LINALG_INV:
        # inverse matrix
        A = inputs[0]
        if isinstance(A, pd.DataFrame):
            A = A.values
        regularize = expr.kwargs.get("regularize", 0.0)
        if regularize > 0:
            A = A + np.eye(A.shape[0]) * regularize
        return np.linalg.inv(A)

    elif expr.op == OpCode.LINALG_PINV:
        # pseudo-inverse (Moore-Penrose)
        A = inputs[0]
        if isinstance(A, pd.DataFrame):
            A = A.values
        rcond = expr.kwargs.get("rcond", None)
        return np.linalg.pinv(A, rcond=rcond)

    elif expr.op == OpCode.LINALG_SOLVE:
        # solve Ax = b
        A, b = inputs[0], inputs[1]
        if isinstance(A, pd.DataFrame):
            A = A.values
        if isinstance(b, (pd.DataFrame, pd.Series)):
            b = b.values
        return np.linalg.solve(A, b)

    elif expr.op == OpCode.LINALG_LSTSQ:
        # least squares solution (OLS): min ||Ax - b||^2
        A, b = inputs[0], inputs[1]
        if isinstance(A, pd.DataFrame):
            A = A.values
        if isinstance(b, (pd.DataFrame, pd.Series)):
            b = b.values
        rcond = expr.kwargs.get("rcond", None)
        result = np.linalg.lstsq(A, b, rcond=rcond)
        return_residuals = expr.kwargs.get("return_residuals", False)
        if return_residuals:
            return {"x": result[0], "residuals": result[1], "rank": result[2], "s": result[3]}
        return result[0]  # return coefficients by default

    # --- Cross-Alpha Operations (for Combiner) ---
    elif expr.op == OpCode.CA_STACK:
        # stack multiple alphas as 3D tensor: (T, N, K) where K = number of alphas
        # inputs is a list of DataFrames (T x N)
        alphas = inputs
        stacked = np.stack([a.values if isinstance(a, pd.DataFrame) else a for a in alphas], axis=-1)
        return stacked

    elif expr.op == OpCode.CA_WEIGHTED_SUM:
        # weighted sum: sum(alpha_k * w_k)
        weights = expr.kwargs.get("weights")  # Dict[str, float] or List[float]
        if isinstance(weights, dict):
            weights = list(weights.values())
        weights = np.array(weights)
        weights = weights / weights.sum()  # normalize

        # inputs[0] is 3D stack or list of DataFrames
        if isinstance(inputs[0], np.ndarray) and inputs[0].ndim == 3:
            # (T, N, K) shape
            result = np.tensordot(inputs[0], weights, axes=([-1], [0]))
            return result
        else:
            # list of DataFrames
            result = None
            for i, alpha in enumerate(inputs):
                if isinstance(alpha, pd.DataFrame):
                    weighted = alpha * weights[i]
                else:
                    weighted = alpha * weights[i]
                if result is None:
                    result = weighted
                else:
                    result = result + weighted
            return result

    elif expr.op == OpCode.CA_RANK_AVERAGE:
        # rank each alpha then average
        weights = expr.kwargs.get("weights")
        if weights is None:
            weights = np.ones(len(inputs)) / len(inputs)
        elif isinstance(weights, dict):
            weights = np.array(list(weights.values()))
            weights = weights / weights.sum()
        else:
            weights = np.array(weights)
            weights = weights / weights.sum()

        ranked_sum = None
        for i, alpha in enumerate(inputs):
            if isinstance(alpha, pd.DataFrame):
                ranked = alpha.rank(axis=1, pct=True)
            else:
                ranked = pd.DataFrame(alpha).rank(axis=1, pct=True).values
            weighted = ranked * weights[i]
            if ranked_sum is None:
                ranked_sum = weighted
            else:
                ranked_sum = ranked_sum + weighted
        return ranked_sum

    elif expr.op == OpCode.CA_IC_WEIGHT:
        # IC-based dynamic weighting
        # inputs: [alphas..., returns]
        returns = inputs[-1]  # last input is returns
        alphas = inputs[:-1]
        lookback = expr.kwargs.get("lookback", 20)
        min_periods = expr.kwargs.get("min_periods", 5)

        ic_weights = []
        for alpha in alphas:
            if isinstance(alpha, pd.DataFrame):
                # correlation between alpha and future returns at each time
                ic = alpha.corrwith(returns.shift(-1), axis=1)
                ic_rolling = ic.rolling(lookback, min_periods=min_periods).mean()
                ic_weights.append(ic_rolling.fillna(0))
            else:
                ic_weights.append(np.zeros(len(returns)))

        # convert IC to weights (make positive + normalize)
        ic_stack = np.stack([w.values if hasattr(w, 'values') else w for w in ic_weights], axis=-1)
        ic_stack = np.maximum(ic_stack, 0)  # negative IC becomes 0
        ic_sum = ic_stack.sum(axis=-1, keepdims=True)
        ic_sum = np.where(ic_sum == 0, 1, ic_sum)  # prevent division by zero
        normalized_weights = ic_stack / ic_sum

        # weighted average
        alpha_stack = np.stack([a.values if isinstance(a, pd.DataFrame) else a for a in alphas], axis=-1)
        result = (alpha_stack * normalized_weights).sum(axis=-1)
        return result

    elif expr.op == OpCode.CA_CORR:
        # compute inter-alpha correlation matrix
        alphas = inputs
        n_alphas = len(alphas)
        corr_matrix = np.zeros((n_alphas, n_alphas))

        for i in range(n_alphas):
            for j in range(n_alphas):
                a_i = alphas[i].values.flatten() if isinstance(alphas[i], pd.DataFrame) else alphas[i].flatten()
                a_j = alphas[j].values.flatten() if isinstance(alphas[j], pd.DataFrame) else alphas[j].flatten()
                # remove NaN
                mask = ~(np.isnan(a_i) | np.isnan(a_j))
                if mask.sum() > 1:
                    corr_matrix[i, j] = np.corrcoef(a_i[mask], a_j[mask])[0, 1]
                else:
                    corr_matrix[i, j] = np.nan

        return corr_matrix

    # --- Cross-Alpha Reduce Operations ---
    elif expr.op == OpCode.CA_REDUCE_AVG:
        alphas = inputs
        threshold = expr.kwargs.get("threshold", 0)
        # Stack alphas and compute mean
        alpha_stack = np.stack([a.values if isinstance(a, pd.DataFrame) else a for a in alphas], axis=-1)
        if threshold > 0:
            # Only average positions where at least `threshold` alphas have valid values
            valid_count = np.sum(~np.isnan(alpha_stack), axis=-1)
            result = np.where(valid_count >= threshold, np.nanmean(alpha_stack, axis=-1), np.nan)
        else:
            result = np.nanmean(alpha_stack, axis=-1)
        if isinstance(alphas[0], pd.DataFrame):
            return pd.DataFrame(result, index=alphas[0].index, columns=alphas[0].columns)
        return result

    elif expr.op == OpCode.CA_REDUCE_SUM:
        alphas = inputs
        alpha_stack = np.stack([a.values if isinstance(a, pd.DataFrame) else a for a in alphas], axis=-1)
        result = np.nansum(alpha_stack, axis=-1)
        if isinstance(alphas[0], pd.DataFrame):
            return pd.DataFrame(result, index=alphas[0].index, columns=alphas[0].columns)
        return result

    elif expr.op == OpCode.CA_REDUCE_MAX:
        alphas = inputs
        alpha_stack = np.stack([a.values if isinstance(a, pd.DataFrame) else a for a in alphas], axis=-1)
        result = np.nanmax(alpha_stack, axis=-1)
        if isinstance(alphas[0], pd.DataFrame):
            return pd.DataFrame(result, index=alphas[0].index, columns=alphas[0].columns)
        return result

    elif expr.op == OpCode.CA_REDUCE_MIN:
        alphas = inputs
        alpha_stack = np.stack([a.values if isinstance(a, pd.DataFrame) else a for a in alphas], axis=-1)
        result = np.nanmin(alpha_stack, axis=-1)
        if isinstance(alphas[0], pd.DataFrame):
            return pd.DataFrame(result, index=alphas[0].index, columns=alphas[0].columns)
        return result

    elif expr.op == OpCode.CA_REDUCE_STDDEV:
        alphas = inputs
        threshold = expr.kwargs.get("threshold", 0)
        alpha_stack = np.stack([a.values if isinstance(a, pd.DataFrame) else a for a in alphas], axis=-1)
        if threshold > 0:
            valid_count = np.sum(~np.isnan(alpha_stack), axis=-1)
            result = np.where(valid_count >= threshold, np.nanstd(alpha_stack, axis=-1), np.nan)
        else:
            result = np.nanstd(alpha_stack, axis=-1)
        if isinstance(alphas[0], pd.DataFrame):
            return pd.DataFrame(result, index=alphas[0].index, columns=alphas[0].columns)
        return result

    elif expr.op == OpCode.CA_REDUCE_IR:
        # IR = mean / std (Information Ratio across alphas)
        alphas = inputs
        alpha_stack = np.stack([a.values if isinstance(a, pd.DataFrame) else a for a in alphas], axis=-1)
        mean = np.nanmean(alpha_stack, axis=-1)
        std = np.nanstd(alpha_stack, axis=-1)
        result = np.where(std != 0, mean / std, np.nan)
        if isinstance(alphas[0], pd.DataFrame):
            return pd.DataFrame(result, index=alphas[0].index, columns=alphas[0].columns)
        return result

    elif expr.op == OpCode.CA_REDUCE_SKEWNESS:
        from scipy import stats as scipy_stats
        alphas = inputs
        alpha_stack = np.stack([a.values if isinstance(a, pd.DataFrame) else a for a in alphas], axis=-1)
        # Compute skewness along last axis
        result = scipy_stats.skew(alpha_stack, axis=-1, nan_policy='omit')
        if isinstance(alphas[0], pd.DataFrame):
            return pd.DataFrame(result, index=alphas[0].index, columns=alphas[0].columns)
        return result

    elif expr.op == OpCode.CA_REDUCE_KURTOSIS:
        from scipy import stats as scipy_stats
        alphas = inputs
        alpha_stack = np.stack([a.values if isinstance(a, pd.DataFrame) else a for a in alphas], axis=-1)
        # Compute kurtosis along last axis
        result = scipy_stats.kurtosis(alpha_stack, axis=-1, nan_policy='omit')
        if isinstance(alphas[0], pd.DataFrame):
            return pd.DataFrame(result, index=alphas[0].index, columns=alphas[0].columns)
        return result

    elif expr.op == OpCode.CA_REDUCE_RANGE:
        alphas = inputs
        alpha_stack = np.stack([a.values if isinstance(a, pd.DataFrame) else a for a in alphas], axis=-1)
        result = np.nanmax(alpha_stack, axis=-1) - np.nanmin(alpha_stack, axis=-1)
        if isinstance(alphas[0], pd.DataFrame):
            return pd.DataFrame(result, index=alphas[0].index, columns=alphas[0].columns)
        return result

    elif expr.op == OpCode.CA_REDUCE_MEDIAN:
        alphas = inputs
        alpha_stack = np.stack([a.values if isinstance(a, pd.DataFrame) else a for a in alphas], axis=-1)
        result = np.nanmedian(alpha_stack, axis=-1)
        if isinstance(alphas[0], pd.DataFrame):
            return pd.DataFrame(result, index=alphas[0].index, columns=alphas[0].columns)
        return result

    elif expr.op == OpCode.CA_REDUCE_COUNT:
        alphas = inputs
        threshold = expr.kwargs.get("threshold", 0)
        alpha_stack = np.stack([a.values if isinstance(a, pd.DataFrame) else a for a in alphas], axis=-1)
        if threshold > 0:
            # Count how many alphas exceed threshold
            result = np.sum(alpha_stack > threshold, axis=-1)
        else:
            # Count non-NaN values
            result = np.sum(~np.isnan(alpha_stack), axis=-1)
        if isinstance(alphas[0], pd.DataFrame):
            return pd.DataFrame(result, index=alphas[0].index, columns=alphas[0].columns)
        return result

    elif expr.op == OpCode.CA_REDUCE_NORM:
        # L2 norm across alphas
        alphas = inputs
        alpha_stack = np.stack([a.values if isinstance(a, pd.DataFrame) else a for a in alphas], axis=-1)
        result = np.sqrt(np.nansum(alpha_stack ** 2, axis=-1))
        if isinstance(alphas[0], pd.DataFrame):
            return pd.DataFrame(result, index=alphas[0].index, columns=alphas[0].columns)
        return result

    elif expr.op == OpCode.CA_REDUCE_POWERSUM:
        alphas = inputs
        power = expr.kwargs.get("power", 2)
        alpha_stack = np.stack([a.values if isinstance(a, pd.DataFrame) else a for a in alphas], axis=-1)
        result = np.nansum(np.abs(alpha_stack) ** power, axis=-1)
        if isinstance(alphas[0], pd.DataFrame):
            return pd.DataFrame(result, index=alphas[0].index, columns=alphas[0].columns)
        return result

    elif expr.op == OpCode.CA_COMBO_A:
        # IR-weighted combination (dynamic weighting based on rolling IC)
        alphas = inputs
        lookback = expr.kwargs.get("lookback", 250)
        mode = expr.kwargs.get("mode", "algo1")

        # For now, use equal weighting if no returns provided
        # Full implementation requires returns data for IC calculation
        alpha_stack = np.stack([a.values if isinstance(a, pd.DataFrame) else a for a in alphas], axis=-1)

        if mode == "algo1":
            # Simple equal weight as fallback
            result = np.nanmean(alpha_stack, axis=-1)
        else:
            result = np.nanmean(alpha_stack, axis=-1)

        if isinstance(alphas[0], pd.DataFrame):
            return pd.DataFrame(result, index=alphas[0].index, columns=alphas[0].columns)
        return result

    # --- Time-Series Additional Operations ---
    elif expr.op == OpCode.TS_DAYS_FROM_LAST_CHANGE:
        x = inputs[0]
        if isinstance(x, pd.DataFrame):
            result = x.copy()
            for col in x.columns:
                changed = x[col].diff().ne(0)
                groups = changed.cumsum()
                result[col] = x.groupby(groups).cumcount()
            return result
        else:
            changed = x.diff().ne(0)
            groups = changed.cumsum()
            return x.groupby(groups).cumcount()

    elif expr.op == OpCode.TS_BACKFILL:
        x = inputs[0]
        limit = expr.kwargs.get("limit", None)
        if isinstance(x, pd.DataFrame):
            return x.bfill(limit=limit)
        else:
            return x.bfill(limit=limit)

    elif expr.op == OpCode.TS_DECAY_EXP_WINDOW:
        x = inputs[0]
        window = expr.kwargs.get("window", 20)
        factor = expr.kwargs.get("factor", 0.5)
        # Exponential decay weights
        weights = np.array([factor ** i for i in range(window)])[::-1]
        weights = weights / weights.sum()

        def apply_decay(series):
            return series.rolling(window, min_periods=1).apply(
                lambda vals: np.sum(vals * weights[-len(vals):] / weights[-len(vals):].sum()),
                raw=True
            )

        if isinstance(x, pd.DataFrame):
            return x.apply(apply_decay)
        else:
            return apply_decay(x)

    elif expr.op == OpCode.TS_HUMP:
        # Detect hump patterns (local maxima)
        x = inputs[0]
        window = expr.kwargs.get("window", 3)

        def detect_hump(series):
            result = pd.Series(0, index=series.index)
            for i in range(window, len(series) - window):
                if series.iloc[i] == series.iloc[i-window:i+window+1].max():
                    result.iloc[i] = 1
            return result

        if isinstance(x, pd.DataFrame):
            return x.apply(detect_hump)
        else:
            return detect_hump(x)

    elif expr.op == OpCode.TS_JUMP_DECAY:
        # Jump detection with decay
        x = inputs[0]
        threshold = expr.kwargs.get("threshold", 2.0)
        decay = expr.kwargs.get("decay", 0.9)

        def jump_decay(series):
            result = pd.Series(0.0, index=series.index)
            returns = series.pct_change()
            std = returns.rolling(20).std()
            jumps = (returns.abs() > threshold * std).astype(float)

            decay_val = 0
            for i in range(len(series)):
                if jumps.iloc[i] == 1:
                    decay_val = 1
                else:
                    decay_val *= decay
                result.iloc[i] = decay_val
            return result

        if isinstance(x, pd.DataFrame):
            return x.apply(jump_decay)
        else:
            return jump_decay(x)

    elif expr.op == OpCode.TS_STEP_DECAY:
        # Step decay
        x = inputs[0]
        step = expr.kwargs.get("step", 5)

        def step_decay(series):
            result = pd.Series(0.0, index=series.index)
            for i in range(len(series)):
                result.iloc[i] = 1.0 / (1 + (i % step))
            return result

        if isinstance(x, pd.DataFrame):
            return x.apply(step_decay)
        else:
            return step_decay(x)

    # --- Transformational Operations ---
    elif expr.op == OpCode.TRADE_WHEN:
        # Only trade (non-zero output) when condition is true
        x = inputs[0]
        condition = inputs[1] if len(inputs) > 1 else expr.kwargs.get("condition")

        if isinstance(x, pd.DataFrame):
            if isinstance(condition, pd.DataFrame):
                return x.where(condition.astype(bool), 0)
            else:
                return x.where(condition, 0)
        else:
            return x.where(condition, 0)

    elif expr.op == OpCode.DENSIFY:
        # Fill sparse data to make it dense
        x = inputs[0]
        method = expr.kwargs.get("method", "ffill")

        if isinstance(x, pd.DataFrame):
            if method == "ffill":
                return x.ffill()
            elif method == "bfill":
                return x.bfill()
            elif method == "interpolate":
                return x.interpolate()
            else:
                return x.ffill()
        else:
            if method == "ffill":
                return x.ffill()
            elif method == "bfill":
                return x.bfill()
            elif method == "interpolate":
                return x.interpolate()
            else:
                return x.ffill()

    # --- Special Operations ---
    elif expr.op == OpCode.SELF_CORR:
        # Auto-correlation with specified lag
        x = inputs[0]
        lag = expr.kwargs.get("lag", 1)

        if isinstance(x, pd.DataFrame):
            result = x.copy()
            for col in x.columns:
                result[col] = x[col].rolling(window=lag+20).apply(
                    lambda vals: pd.Series(vals).autocorr(lag=lag), raw=False
                )
            return result
        else:
            return x.rolling(window=lag+20).apply(
                lambda vals: pd.Series(vals).autocorr(lag=lag), raw=False
            )

    elif expr.op == OpCode.VECTOR_NEUT:
        # Neutralize against a vector (orthogonalization)
        x = inputs[0]
        vector = inputs[1] if len(inputs) > 1 else expr.kwargs.get("vector")

        def neutralize_against(data, vec):
            # Project out the component along vec
            if isinstance(data, pd.DataFrame):
                result = data.copy()
                for idx in data.index:
                    row = data.loc[idx].values
                    v = vec.loc[idx].values if isinstance(vec, pd.DataFrame) else vec
                    # Orthogonalize: x - (x·v/v·v) * v
                    dot_xv = np.nansum(row * v)
                    dot_vv = np.nansum(v * v)
                    if dot_vv != 0:
                        result.loc[idx] = row - (dot_xv / dot_vv) * v
                return result
            else:
                dot_xv = np.nansum(data.values * vector.values)
                dot_vv = np.nansum(vector.values ** 2)
                if dot_vv != 0:
                    return data - (dot_xv / dot_vv) * vector
                return data

        return neutralize_against(x, vector)

    elif expr.op == OpCode.CLIP_EXTREME:
        x = inputs[0]
        n_std = expr.kwargs.get("n_std", 3.0)

        if isinstance(x, pd.DataFrame):
            result = x.copy()
            for timestamp in x.index:
                row = x.loc[timestamp]
                mean = row.mean()
                std = row.std()
                if std > 0:
                    lower = mean - n_std * std
                    upper = mean + n_std * std
                    result.loc[timestamp] = row.clip(lower=lower, upper=upper)
            return result
        else:
            mean = x.mean()
            std = x.std()
            if std > 0:
                lower = mean - n_std * std
                upper = mean + n_std * std
                return x.clip(lower=lower, upper=upper)
            return x

    else:
        raise NotImplementedError(f"Operator not implemented: {expr.op}")


# --- Strata Helper Functions ---

def _by_operation(
    data: pd.DataFrame,
    strata: Dict[str, str],
    operation: str,
    **kwargs
) -> pd.DataFrame:
    """Apply operation within each strata category.

    Args:
        data: DataFrame (T x N) with symbols as columns
        strata: Dict mapping symbol -> category
        operation: Operation name ("demean", "rank", "zscore", "scale", "mean", etc.)
        **kwargs: Additional arguments for the operation

    Returns:
        DataFrame with operation applied within each strata
    """
    if isinstance(data, pd.Series):
        # Handle Series case
        return _by_operation_series(data, strata, operation, **kwargs)

    result = data.copy()

    for timestamp in data.index:
        row = data.loc[timestamp]

        # Group symbols by strata category
        strata_groups: Dict[str, List[str]] = {}
        for symbol in row.index:
            if pd.notna(row[symbol]):
                category = strata.get(symbol, "Unknown")
                if category not in strata_groups:
                    strata_groups[category] = []
                strata_groups[category].append(symbol)

        # Apply operation within each strata
        for category, symbols in strata_groups.items():
            if len(symbols) == 0:
                continue

            values = row[symbols]

            if operation == "demean":
                result.loc[timestamp, symbols] = values - values.mean()

            elif operation == "rank":
                pct = kwargs.get("pct", True)
                result.loc[timestamp, symbols] = values.rank(pct=pct)

            elif operation == "zscore":
                mean = values.mean()
                std = values.std()
                if std == 0 or pd.isna(std):
                    result.loc[timestamp, symbols] = 0
                else:
                    result.loc[timestamp, symbols] = (values - mean) / std

            elif operation == "scale":
                scale_val = kwargs.get("scale_val", 1.0)
                abs_sum = values.abs().sum()
                if abs_sum == 0 or pd.isna(abs_sum):
                    result.loc[timestamp, symbols] = 0
                else:
                    result.loc[timestamp, symbols] = values / abs_sum * scale_val

            elif operation == "mean":
                result.loc[timestamp, symbols] = values.mean()

            elif operation == "sum":
                result.loc[timestamp, symbols] = values.sum()

            elif operation == "max":
                result.loc[timestamp, symbols] = values.max()

            elif operation == "min":
                result.loc[timestamp, symbols] = values.min()

            elif operation == "std":
                result.loc[timestamp, symbols] = values.std()

            elif operation == "median":
                result.loc[timestamp, symbols] = values.median()

            elif operation == "count":
                result.loc[timestamp, symbols] = values.count()

    return result


def _by_operation_series(
    data: pd.Series,
    strata: Dict[str, str],
    operation: str,
    **kwargs
) -> pd.Series:
    """Apply operation within each strata category for Series."""
    result = data.copy()

    # Group symbols by strata category
    strata_groups: Dict[str, List[str]] = {}
    for symbol, value in data.items():
        if pd.notna(value):
            category = strata.get(symbol, "Unknown")
            if category not in strata_groups:
                strata_groups[category] = []
            strata_groups[category].append(symbol)

    # Apply operation within each strata
    for category, symbols in strata_groups.items():
        if len(symbols) == 0:
            continue

        values = data[symbols]

        if operation == "demean":
            result[symbols] = values - values.mean()

        elif operation == "rank":
            pct = kwargs.get("pct", True)
            result[symbols] = values.rank(pct=pct)

        elif operation == "zscore":
            mean = values.mean()
            std = values.std()
            if std == 0 or pd.isna(std):
                result[symbols] = 0
            else:
                result[symbols] = (values - mean) / std

        elif operation == "scale":
            scale_val = kwargs.get("scale_val", 1.0)
            abs_sum = values.abs().sum()
            if abs_sum == 0 or pd.isna(abs_sum):
                result[symbols] = 0
            else:
                result[symbols] = values / abs_sum * scale_val

        elif operation == "mean":
            result[symbols] = values.mean()

        elif operation == "sum":
            result[symbols] = values.sum()

        elif operation == "max":
            result[symbols] = values.max()

        elif operation == "min":
            result[symbols] = values.min()

        elif operation == "std":
            result[symbols] = values.std()

        elif operation == "median":
            result[symbols] = values.median()

        elif operation == "count":
            result[symbols] = values.count()

    return result


def _binning(
    data: Union[pd.DataFrame, pd.Series],
    n_bins: int = 5,
    labels: Optional[List[str]] = None,
) -> Union[pd.DataFrame, pd.Series]:
    """Create strata from data values using quantile binning.

    Args:
        data: Input data
        n_bins: Number of bins (default: 5 for quintiles)
        labels: Optional custom labels for bins

    Returns:
        DataFrame/Series with bin labels as values

    Example:
        >>> # Create quintile bins from returns
        >>> return_bins = binning(returns, n_bins=5)
        >>> # Use as strata for other operations
        >>> by_demean(alpha, strata=return_bins)
    """
    if labels is None:
        labels = [f"bin_{i+1}" for i in range(n_bins)]

    if len(labels) != n_bins:
        raise ValueError(f"Number of labels ({len(labels)}) must match n_bins ({n_bins})")

    if isinstance(data, pd.DataFrame):
        result = pd.DataFrame(index=data.index, columns=data.columns)
        for col in data.columns:
            try:
                result[col] = pd.qcut(data[col], q=n_bins, labels=labels, duplicates="drop")
            except ValueError:
                # Handle case where there are too few unique values
                result[col] = labels[0]
        return result
    else:
        try:
            return pd.qcut(data, q=n_bins, labels=labels, duplicates="drop")
        except ValueError:
            return pd.Series(labels[0], index=data.index)


# --- Operator Wrapper ---

def wrap_operator(op_code: OpCode):
    """Decorator to wrap function as Expr-returning operator."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(data, *args, **kwargs):
            if isinstance(data, Expr):
                return Expr(op_code, args=args, kwargs=kwargs, inputs=[data])
            expr = Expr.from_data(data)
            result_expr = Expr(op_code, args=args, kwargs=kwargs, inputs=[expr])
            return result_expr.execute()

        wrapper._op_code = op_code
        wrapper._original_func = func
        return wrapper
    return decorator


# --- Custom Operator Base Class ---

class BaseOperator(ABC):
    """Base class for custom operators."""

    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def to_expr(self, data: Expr, *args, **kwargs) -> Expr:
        """Build expression tree for this operator."""
        pass

    def __call__(self, data, *args, **kwargs) -> Expr:
        """Apply operator to data."""
        if not isinstance(data, Expr):
            data = Expr.from_data(data)
        return self.to_expr(data, *args, **kwargs)


# --- Operator Registry ---

_OPERATOR_REGISTRY: Dict[str, type] = {}


def register_operator(name: str):
    """Decorator to register custom operators."""
    def decorator(cls: type) -> type:
        if not issubclass(cls, BaseOperator):
            raise TypeError(f"{cls.__name__} must inherit from BaseOperator")
        _OPERATOR_REGISTRY[name] = cls
        return cls
    return decorator


def get_operator(name: str) -> Optional[type]:
    """Get registered operator by name."""
    return _OPERATOR_REGISTRY.get(name)


def list_operators() -> List[str]:
    """List all registered operators."""
    return list(_OPERATOR_REGISTRY.keys())


# --- ExprDataFrame - DataFrame Proxy ---

class ExprDataFrame:
    """DataFrame wrapper that returns Expr on attribute access."""

    def __init__(self, data: pd.DataFrame):
        self._data = data

    def __getattr__(self, name: str) -> Expr:
        if name.startswith("_"):
            raise AttributeError(name)
        return Expr.col(name)

    def __getitem__(self, key: str) -> Expr:
        return Expr.col(key)

    def execute(self, expr: Expr) -> Union[pd.DataFrame, pd.Series]:
        """Execute expression with this DataFrame."""
        return expr.execute(self._data)
