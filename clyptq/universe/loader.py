"""Universe data loader for analysis.

This module provides utilities to load and prepare data from a Universe
for use in Jupyter notebooks and alpha research.

NEW DESIGN (in_universe column-based):
- Input is ALWAYS N (all symbols in data_store) for warmup/time-series continuity
- Universe is a MASK, not a data filter
- in_universe column tracks which symbols are in target universe
- scope parameter controls COMPUTATION scope (LOCAL=n, GLOBAL=N)
- output parameter controls OUTPUT scope (TRUNCATED=n, FULL=N)

Usage:
    ```python
    from clyptq.universe import StaticUniverse, load_universe
    from clyptq.data import DataStore
    from clyptq import operator

    # Define target universe
    target = StaticUniverse(["BTCUSDT", "ETHUSDT"])  # n = 2

    # Load data (always loads N, but marks in_universe)
    data = load_universe(target, store)

    # data.columns = all N symbols
    # data['in_universe'] row = Boolean mask for each timestamp

    # Cross-section operator use in_universe for scope control
    alpha = operator.rank(data['close'], in_universe=data['in_universe'], scope="LOCAL")
    ```
"""

from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Set, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from clyptq.data.sources import DataSource
    from clyptq.universe.base import BaseUniverse


FieldType = Literal["open", "high", "low", "close", "volume", "returns", "log_returns"]
ScopeType = Literal["LOCAL", "GLOBAL"]
OutputType = Literal["TRUNCATED", "FULL"]


class DataIntegrityError(ValueError):
    """Error raised when universe symbols have missing data."""
    pass


class UniverseData:
    """Container for universe data with in_universe mask.

    Holds wide-format DataFrames (T x N) with an in_universe mask DataFrame
    that tracks which symbols are in the target universe at each timestamp.

    Attributes:
        open: Open prices (T x N)
        high: High prices (T x N)
        low: Low prices (T x N)
        close: Close prices (T x N)
        volume: Volume (T x N)
        returns: Simple returns (T x N)
        in_universe: Boolean mask (T x N) - True if symbol in target universe
        target_symbols: Set of target universe symbols (n)
        all_symbols: List of all symbols (N)

    Example:
        ```python
        data = load_universe(universe, store)

        # Access price data
        close = data.close  # (T x N) DataFrame

        # Access universe mask
        mask = data.in_universe  # (T x N) Boolean DataFrame

        # Check for data errors (universe symbols with NaN)
        errors = data.data_errors()  # List of (timestamp, symbol) with errors

        # Get only universe symbols
        close_n = data.get_universe_only('close')  # (T x n) DataFrame
        ```
    """

    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        in_universe: pd.DataFrame,
        target_symbols: Set[str],
        all_symbols: List[str],
    ):
        """Initialize UniverseData.

        Args:
            data: Dict of field name -> DataFrame (T x N)
            in_universe: Boolean DataFrame (T x N)
            target_symbols: Set of target universe symbols
            all_symbols: List of all symbols
        """
        self._data = data
        self._in_universe = in_universe
        self._target_symbols = target_symbols
        self._all_symbols = all_symbols

    @property
    def open(self) -> pd.DataFrame:
        """Open prices (T x N)."""
        return self._data.get('open', pd.DataFrame())

    @property
    def high(self) -> pd.DataFrame:
        """High prices (T x N)."""
        return self._data.get('high', pd.DataFrame())

    @property
    def low(self) -> pd.DataFrame:
        """Low prices (T x N)."""
        return self._data.get('low', pd.DataFrame())

    @property
    def close(self) -> pd.DataFrame:
        """Close prices (T x N)."""
        return self._data.get('close', pd.DataFrame())

    @property
    def volume(self) -> pd.DataFrame:
        """Volume (T x N)."""
        return self._data.get('volume', pd.DataFrame())

    @property
    def returns(self) -> pd.DataFrame:
        """Simple returns (T x N)."""
        return self._data.get('returns', pd.DataFrame())

    @property
    def log_returns(self) -> pd.DataFrame:
        """Log returns (T x N)."""
        return self._data.get('log_returns', pd.DataFrame())

    @property
    def in_universe(self) -> pd.DataFrame:
        """Universe membership mask (T x N)."""
        return self._in_universe

    @property
    def target_symbols(self) -> Set[str]:
        """Target universe symbols (n)."""
        return self._target_symbols

    @property
    def all_symbols(self) -> List[str]:
        """All symbols (N)."""
        return self._all_symbols

    @property
    def n_symbols(self) -> int:
        """Number of target universe symbols."""
        return len(self._target_symbols)

    @property
    def N_symbols(self) -> int:
        """Total number of symbols."""
        return len(self._all_symbols)

    def __getitem__(self, field: str) -> pd.DataFrame:
        """Get data by field name (dict-like access).

        Args:
            field: One of 'open', 'high', 'low', 'close', 'volume', 'returns', 'log_returns'

        Returns:
            DataFrame (T x N)

        Example:
            ```python
            data = load_universe(universe, store)
            close = data["close"]  # Same interface as DataProvider
            ```
        """
        if field not in self._data:
            raise KeyError(f"Unknown field: {field}. Available: {list(self._data.keys())}")
        return self._data[field]

    def get(self, field: str, default=None) -> pd.DataFrame:
        """Get data by field name with default.

        Args:
            field: One of 'open', 'high', 'low', 'close', 'volume', 'returns', 'log_returns'
            default: Default value if field not found

        Returns:
            DataFrame (T x N) or default
        """
        try:
            return self[field]
        except KeyError:
            return default

    def get_universe_only(self, field: str) -> pd.DataFrame:
        """Get data filtered to target universe symbols only.

        Args:
            field: Field name

        Returns:
            DataFrame (T x n) with only target universe columns
        """
        df = self.get(field)
        cols = [c for c in df.columns if c in self._target_symbols]
        return df[cols]

    def data_errors(self) -> List[tuple]:
        """Find data errors: universe symbols with NaN values.

        Returns:
            List of (timestamp, symbol) tuples where in_universe=True but value is NaN
        """
        errors = []
        close = self.close

        for ts in close.index:
            for symbol in close.columns:
                if self._in_universe.loc[ts, symbol]:  # In universe
                    if pd.isna(close.loc[ts, symbol]):  # But has NaN
                        errors.append((ts, symbol))

        return errors

    def has_data_errors(self) -> bool:
        """Check if any data errors exist."""
        return len(self.data_errors()) > 0

    def validate(self, raise_on_error: bool = True) -> bool:
        """Validate data integrity.

        Args:
            raise_on_error: If True, raise DataIntegrityError on errors

        Returns:
            True if valid, False otherwise

        Raises:
            DataIntegrityError: If raise_on_error and errors found
        """
        errors = self.data_errors()
        if errors:
            if raise_on_error:
                sample = errors[:5]
                raise DataIntegrityError(
                    f"Found {len(errors)} data errors (universe symbols with NaN). "
                    f"Sample: {sample}"
                )
            return False
        return True

    def __repr__(self) -> str:
        return (
            f"UniverseData(N={self.N_symbols}, n={self.n_symbols}, "
            f"T={len(self.close)}, fields={list(self._data.keys())})"
        )


def load_universe(
    universe: "BaseUniverse",
    data_store: "DataStore",
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    timestamp: Optional[datetime] = None,
    validate: bool = True,
) -> UniverseData:
    """Load data with universe membership tracking.

    ALWAYS loads ALL symbols (N) from data_store. Universe membership
    is tracked via in_universe mask, not by filtering symbols.

    This ensures:
    1. Warmup data available for symbols entering universe
    2. Time-series continuity for all symbols
    3. Clear separation of data availability vs universe membership

    Args:
        universe: Universe object defining target symbols (n)
        data_store: DataStore containing the OHLCV data
        start: Start timestamp (defaults to earliest available)
        end: End timestamp (defaults to latest available)
        timestamp: Reference timestamp for universe membership determination
        validate: If True, raise error if universe symbols have missing data

    Returns:
        UniverseData container with:
            - OHLCV DataFrames (T x N)
            - in_universe mask (T x N)
            - target_symbols set

    Raises:
        DataIntegrityError: If validate=True and universe symbols have NaN

    Example:
        ```python
        # Load data
        data = load_universe(universe, store)

        # Use in operator
        from clyptq import operator

        # Rank within universe (LOCAL scope)
        alpha = operator.rank(data.close, in_universe=data.in_universe, scope="LOCAL")

        # Rank across all symbols (GLOBAL scope)
        alpha = operator.rank(data.close, scope="GLOBAL")
        ```
    """
    # Determine date range
    data_range = data_store.get_date_range()
    if start is None:
        start = data_range.start
    if end is None:
        end = data_range.end

    # Get reference timestamp for universe membership
    if timestamp is None:
        timestamp = end

    # Get target universe symbols (n)
    target_symbols = set(universe.get_symbols(timestamp, data_store))

    if not target_symbols:
        raise ValueError(f"No symbols in target universe at {timestamp}")

    # ALWAYS load ALL symbols (N) - this is the key design change
    all_symbols = sorted(data_store.symbols())

    # Build wide DataFrames for each field
    data_dict: Dict[str, Dict[str, pd.Series]] = {
        'open': {},
        'high': {},
        'low': {},
        'close': {},
        'volume': {},
    }

    for symbol in all_symbols:
        if not data_store.has_symbol(symbol):
            continue

        try:
            df = data_store._data[symbol]

            # Filter date range
            mask = (df.index >= start) & (df.index <= end)
            df = df.loc[mask]

            if len(df) == 0:
                continue

            # Store each field
            data_dict['open'][symbol] = df['open']
            data_dict['high'][symbol] = df['high']
            data_dict['low'][symbol] = df['low']
            data_dict['close'][symbol] = df['close']
            data_dict['volume'][symbol] = df['volume']

        except (KeyError, ValueError):
            continue

    if not data_dict['close']:
        raise ValueError("No data loaded for any symbols")

    # Convert to DataFrames
    result_data = {}
    for field, series_dict in data_dict.items():
        df = pd.DataFrame(series_dict)
        df = df.reindex(sorted(df.columns), axis=1)  # Sort columns
        result_data[field] = df

    # Add computed fields
    close_df = result_data['close']
    result_data['returns'] = close_df.pct_change()
    result_data['log_returns'] = np.log(close_df).diff()

    # Build in_universe mask (T x N)
    # For now, use static membership based on reference timestamp
    # TODO: Support dynamic universe membership per timestamp
    in_universe = pd.DataFrame(
        index=close_df.index,
        columns=close_df.columns,
        data=False,
    )
    for symbol in target_symbols:
        if symbol in in_universe.columns:
            in_universe[symbol] = True

    # Create container
    universe_data = UniverseData(
        data=result_data,
        in_universe=in_universe,
        target_symbols=target_symbols,
        all_symbols=list(close_df.columns),
    )

    # Validate if requested
    if validate:
        universe_data.validate(raise_on_error=True)

    return universe_data


def load_universe_dynamic(
    universe: "BaseUniverse",
    data_store: "DataStore",
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    rebalance_freq: str = "D",
) -> UniverseData:
    """Load data with dynamic universe membership (per-timestamp).

    Unlike load_universe which uses a fixed reference timestamp,
    this function computes universe membership at each timestamp,
    supporting time-varying universes.

    Args:
        universe: Universe object (can be DynamicUniverse)
        data_store: DataStore containing the OHLCV data
        start: Start timestamp
        end: End timestamp
        rebalance_freq: How often to recompute universe membership
            - "D": Daily
            - "W": Weekly
            - "M": Monthly

    Returns:
        UniverseData with per-timestamp in_universe mask
    """
    # Determine date range
    data_range = data_store.get_date_range()
    if start is None:
        start = data_range.start
    if end is None:
        end = data_range.end

    # Load all data first
    all_symbols = sorted(data_store.symbols())

    data_dict: Dict[str, Dict[str, pd.Series]] = {
        'open': {}, 'high': {}, 'low': {}, 'close': {}, 'volume': {},
    }

    for symbol in all_symbols:
        if not data_store.has_symbol(symbol):
            continue
        try:
            df = data_store._data[symbol]
            mask = (df.index >= start) & (df.index <= end)
            df = df.loc[mask]
            if len(df) == 0:
                continue
            for field in ['open', 'high', 'low', 'close', 'volume']:
                data_dict[field][symbol] = df[field]
        except (KeyError, ValueError):
            continue

    # Convert to DataFrames
    result_data = {}
    for field, series_dict in data_dict.items():
        df = pd.DataFrame(series_dict)
        df = df.reindex(sorted(df.columns), axis=1)
        result_data[field] = df

    close_df = result_data['close']
    result_data['returns'] = close_df.pct_change()
    result_data['log_returns'] = np.log(close_df).diff()

    # Build dynamic in_universe mask
    in_universe = pd.DataFrame(
        index=close_df.index,
        columns=close_df.columns,
        data=False,
    )

    # Get rebalance dates
    if rebalance_freq == "D":
        rebalance_dates = close_df.index
    elif rebalance_freq == "W":
        rebalance_dates = close_df.resample("W").first().index
    elif rebalance_freq == "M":
        rebalance_dates = close_df.resample("ME").first().index
    else:
        rebalance_dates = close_df.index

    # Compute universe membership at each rebalance date
    current_universe: Set[str] = set()
    last_rebalance_idx = 0

    for i, ts in enumerate(close_df.index):
        # Check if we need to rebalance
        if ts in rebalance_dates or i == 0:
            current_universe = set(universe.get_symbols(ts.to_pydatetime(), data_store))

        # Set membership for this timestamp
        for symbol in current_universe:
            if symbol in in_universe.columns:
                in_universe.loc[ts, symbol] = True

    # Get final target symbols (union of all periods)
    all_target_symbols = set()
    for ts in close_df.index:
        all_target_symbols.update(
            col for col in in_universe.columns if in_universe.loc[ts, col]
        )

    return UniverseData(
        data=result_data,
        in_universe=in_universe,
        target_symbols=all_target_symbols,
        all_symbols=list(close_df.columns),
    )


def filter_to_target(
    data: Union[pd.DataFrame, "UniverseData"],
    target_symbols: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """Filter DataFrame to target universe symbols.

    Args:
        data: DataFrame or UniverseData to filter
        target_symbols: Target symbols to keep

    Returns:
        Filtered DataFrame with only target symbols
    """
    if isinstance(data, UniverseData):
        target_symbols = data.target_symbols
        data = data.close

    if target_symbols is None:
        if hasattr(data, 'target_symbols') and data.target_symbols:
            target_symbols = data.target_symbols
        else:
            raise ValueError("target_symbols must be provided")

    available = set(data.columns) & target_symbols
    return data[sorted(available)]
