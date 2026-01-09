"""
DataProvider: Clock + Data + Universe unified interface.

Architecture:
- DataSource: Internal data loading abstraction (Parquet, Memory, Live)
- DataProvider: Buffer + Clock + Universe management
- Multi-timeframe: System clock = OHLCV timeframe, other data aligned via ffill

Flow:
    User entrypoint → DataProvider → Strategy → Engine

Multi-timeframe Support:
    - System clock = OHLCV timeframe (minimum granularity)
    - Other data timeframes must be >= OHLCV timeframe
    - Access: provider["close"] or provider["close", "4h"] for resampled
    - Alignment: Larger timeframes are forward-filled to system clock

Example:
    ```python
    # Backtest with Parquet files
    provider = DataProvider.from_parquet(
        path="data/crypto/",
        symbols=["BTC", "ETH"],
        rebalance_freq="1d",
        timeframe="1h",  # System clock = 1h
    )
    provider.load(start=start, end=end)

    # Access data
    close = provider["close"]           # 1h (system clock)
    close_4h = provider["close", "4h"]  # Resampled to 4h
    funding = provider["funding_rate"]  # Aligned via ffill

    # Engine runs strategy
    engine = Engine()
    result = engine.run(strategy, mode="backtest", ...)
    ```
"""

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

import pandas as pd

from clyptq.core.clock import BacktestClock, Clock, LiveClock
from clyptq.core.timeframe import (
    minutes_to_timeframe,
    timeframe_to_minutes,
)
from clyptq.data.sources.base import DataSource

if TYPE_CHECKING:
    from clyptq.data.spec import DataSpecType
    from clyptq.universe.base import BaseUniverse


class DataProvider:
    """Unified data provider with clock management.

    Responsibilities:
    - Clock: Time progression via Clock abstraction (system clock = OHLCV timeframe)
    - Data: Provide data up to current timestamp (no look-ahead)
    - Universe: Track in_universe membership per tick
    - Rebalance: Determine rebalancing schedule
    - Buffer: Manage lookback window
    - Multi-timeframe: Resample OHLCV to larger timeframes, align other data via ffill
    - Strata: Provide group/category metadata for symbols (e.g., layer, sector, market_cap_tier)

    Attributes:
        universe: Universe defining tradeable symbols
        sources: Dict of {name: DataSource}
        rebalance_freq: Rebalancing frequency ("1h", "1d", "1w")
        system_clock: OHLCV timeframe (tick interval)
        clock: Clock instance (BacktestClock or LiveClock)
    """

    # OHLCV aggregation rules for resampling
    _OHLCV_AGG_RULES = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    # --- Mock Strata Data ---
    # TODO: Replace with real data from data collection pipeline
    # TODO: Strata should be loaded from external source (DB, API, or file)
    _MOCK_CRYPTO_STRATA = {
        # Layer classification (blockchain layer)
        "layer": {
            "BTC": "L1",
            "ETH": "L1",
            "SOL": "L1",
            "AVAX": "L1",
            "ADA": "L1",
            "DOT": "L1",
            "ATOM": "L1",
            "NEAR": "L1",
            "FTM": "L1",
            "ALGO": "L1",
            "MATIC": "L2",
            "ARB": "L2",
            "OP": "L2",
            "IMX": "L2",
            "MINA": "L2",
            "UNI": "DeFi",
            "AAVE": "DeFi",
            "LINK": "DeFi",
            "MKR": "DeFi",
            "CRV": "DeFi",
            "SNX": "DeFi",
            "COMP": "DeFi",
            "SUSHI": "DeFi",
            "YFI": "DeFi",
            "1INCH": "DeFi",
            "DOGE": "Meme",
            "SHIB": "Meme",
            "PEPE": "Meme",
            "FLOKI": "Meme",
            "BONK": "Meme",
            "BNB": "Exchange",
            "FTT": "Exchange",
            "CRO": "Exchange",
            "OKB": "Exchange",
            "KCS": "Exchange",
            "SAND": "Metaverse",
            "MANA": "Metaverse",
            "AXS": "Metaverse",
            "ENJ": "Metaverse",
            "GALA": "Metaverse",
            "FIL": "Infrastructure",
            "AR": "Infrastructure",
            "THETA": "Infrastructure",
            "RENDER": "Infrastructure",
            "HNT": "Infrastructure",
            "XRP": "Payment",
            "XLM": "Payment",
            "LTC": "Payment",
            "BCH": "Payment",
            "DASH": "Payment",
        },
        # Market cap tier (based on typical rankings)
        "market_cap_tier": {
            "BTC": "mega",
            "ETH": "mega",
            "BNB": "large",
            "XRP": "large",
            "SOL": "large",
            "ADA": "large",
            "DOGE": "large",
            "AVAX": "mid",
            "DOT": "mid",
            "MATIC": "mid",
            "LINK": "mid",
            "SHIB": "mid",
            "LTC": "mid",
            "ATOM": "mid",
            "UNI": "mid",
            "ARB": "small",
            "OP": "small",
            "NEAR": "small",
            "FTM": "small",
            "AAVE": "small",
            "MKR": "small",
            "ALGO": "small",
            "FIL": "small",
            "SAND": "small",
            "MANA": "small",
            "AXS": "small",
            "CRV": "micro",
            "SNX": "micro",
            "COMP": "micro",
            "SUSHI": "micro",
            "YFI": "micro",
            "1INCH": "micro",
            "ENJ": "micro",
            "GALA": "micro",
            "PEPE": "micro",
            "FLOKI": "micro",
            "BONK": "micro",
        },
        # Volatility tier (typical behavior)
        "volatility_tier": {
            "BTC": "low",
            "ETH": "low",
            "BNB": "low",
            "XRP": "medium",
            "SOL": "medium",
            "ADA": "medium",
            "DOGE": "high",
            "AVAX": "medium",
            "DOT": "medium",
            "MATIC": "medium",
            "LINK": "medium",
            "SHIB": "extreme",
            "LTC": "low",
            "UNI": "medium",
            "AAVE": "medium",
            "PEPE": "extreme",
            "FLOKI": "extreme",
            "BONK": "extreme",
            "ARB": "high",
            "OP": "high",
        },
    }

    def __init__(
        self,
        universe: "BaseUniverse" = None,
        sources: Optional[Dict[str, DataSource]] = None,
        rebalance_freq: str = "1d",
        mode: Literal["backtest", "live", "research"] = "backtest",
        clock: Optional[Clock] = None,
        max_bars: Optional[int] = None,
        *,
        data: Optional[Dict[str, "DataSpecType"]] = None,
        specs: Optional[Dict[str, "DataSpecType"]] = None,
    ):
        """Initialize DataProvider.

        Args:
            universe: Universe defining symbols
            sources: Dict of {name: DataSource} (runtime binding)
            rebalance_freq: Rebalancing frequency
            mode: "backtest", "live", or "research"
            clock: Optional Clock instance (created automatically if not provided)
            max_bars: Buffer limit (especially for live mode)
            data: Dict of {name: DataSpec} for declarative usage (Engine binds source at runtime)
            specs: Alias for data (preferred name)

        Note:
            - mode="research": Auto-creates sources from specs, loads all data at once
            - mode="backtest": Engine binds sources at runtime (for tick-by-tick simulation)
            - mode="live": Engine binds live sources

        Example:
            ```python
            # Research mode: auto-creates sources from specs
            p = DataProvider(
                universe=CryptoLiquid(top_n=30),
                specs={"ohlcv": OHLCVSpec(exchange="gateio", market_type="spot")},
                mode="research",
            )
            p.load(start=START, end=END)
            close = p["close"]  # Full DataFrame
            ```
        """
        self.universe = universe
        self.sources = sources or {}
        self._data_specs = specs or data  # specs is preferred alias
        self.rebalance_freq = rebalance_freq
        self.mode = mode
        self._rebalance_minutes = timeframe_to_minutes(rebalance_freq)

        # Research mode: auto-create sources from specs
        if mode == "research" and self._data_specs and not self.sources:
            self.sources = self._build_sources_from_specs(self._data_specs)

        # Buffer limit
        if mode == "live" and max_bars is None:
            import warnings
            warnings.warn(
                "max_bars not specified for live mode. Defaulting to 500.",
                UserWarning,
            )
            max_bars = 500
        self.max_bars = max_bars

        # Build field -> source mapping
        self._field_to_source: Dict[str, str] = {}

        # Declarative mode: sources will be bound later by Engine
        # In this case, we skip validation until bind_sources() is called
        self._is_declarative = (not self.sources) and (self._data_specs is not None)

        if self._is_declarative:
            # Declarative mode: defer validation until Engine binds sources
            self.system_clock = "1d"  # Default, will be updated when sources bound
            self._system_clock_minutes = timeframe_to_minutes(self.system_clock)
            self.clock = None  # Will be created when sources bound
        else:
            # Runtime mode: validate sources immediately
            if not self.sources:
                raise ValueError("At least one source must be provided (or use data= for declarative mode)")

            self._validate_ohlcv_available()

            # System clock = OHLCV timeframe (not GCD)
            ohlcv_source = self.sources.get("ohlcv")
            if ohlcv_source is None:
                # Find first source with OHLCV fields
                for source in self.sources.values():
                    if "close" in source.fields:
                        ohlcv_source = source
                        break

            if ohlcv_source is None:
                raise ValueError("No OHLCV source found")

            self.system_clock = ohlcv_source.timeframe
            self._system_clock_minutes = timeframe_to_minutes(self.system_clock)

            # Validate other data timeframes >= OHLCV timeframe
            self._validate_timeframes()

            # Create or use provided clock
            if clock is not None:
                self.clock = clock
            elif mode == "live":
                self.clock = LiveClock(self.system_clock)
            else:
                self.clock = BacktestClock(self.system_clock)

            # Build field -> source mapping
            for source_name, source in self.sources.items():
                for field in source.fields:
                    if field in self._field_to_source:
                        raise ValueError(
                            f"Field '{field}' defined in multiple sources: "
                            f"{self._field_to_source[field]} and {source_name}"
                        )
                    self._field_to_source[field] = source_name

        # Data cache
        self._data: Dict[str, pd.DataFrame] = {}
        self._resampled_cache: Dict[str, pd.DataFrame] = {}  # Cache for resampled data
        self._source_fields: Dict[str, List[str]] = {}
        self._symbols: List[str] = []

        # Clock state
        self._timestamps: List[datetime] = []
        self._current_idx: int = -1
        self._current_ts: Optional[datetime] = None
        self._last_rebalance_ts: Optional[datetime] = None

        # Universe state
        self._in_universe: pd.Series = pd.Series(dtype=bool)

        self._loaded = False

    def bind_sources(
        self,
        sources: Dict[str, DataSource],
        mode: Literal["backtest", "live"] = "backtest",
    ) -> "DataProvider":
        """Bind sources to a declarative DataProvider.

        Called by Engine to bind sources based on mode (backtest=parquet, live=exchange).

        Args:
            sources: Dict of {name: DataSource}
            mode: "backtest" or "live"

        Returns:
            self for method chaining
        """
        self.sources = sources
        self.mode = mode
        self._is_declarative = False

        # Validate sources
        if not self.sources:
            raise ValueError("At least one source must be provided")

        self._validate_ohlcv_available()

        # System clock = OHLCV timeframe
        ohlcv_source = self.sources.get("ohlcv")
        if ohlcv_source is None:
            for source in self.sources.values():
                if "close" in source.fields:
                    ohlcv_source = source
                    break

        if ohlcv_source is None:
            raise ValueError("No OHLCV source found")

        self.system_clock = ohlcv_source.timeframe
        self._system_clock_minutes = timeframe_to_minutes(self.system_clock)

        # Validate other data timeframes >= OHLCV timeframe
        self._validate_timeframes()

        # Create clock
        if mode == "live":
            self.clock = LiveClock(self.system_clock)
        else:
            self.clock = BacktestClock(self.system_clock)

        # Build field -> source mapping
        self._field_to_source = {}
        for source_name, source in self.sources.items():
            for field in source.fields:
                if field in self._field_to_source:
                    raise ValueError(
                        f"Field '{field}' defined in multiple sources: "
                        f"{self._field_to_source[field]} and {source_name}"
                    )
                self._field_to_source[field] = source_name

        return self

    @property
    def data_specs(self) -> Optional[Dict[str, "DataSpecType"]]:
        """Get data specs (for declarative mode)."""
        return self._data_specs

    def _build_sources_from_specs(self, specs: Dict[str, "DataSpecType"]) -> Dict[str, DataSource]:
        """Build DataSource dict from specs (for research mode).

        Auto-resolves path from OHLCVSpec:
            {PROJECT_ROOT}/data/{market_type}/{exchange}/{timeframe}/

        Args:
            specs: Dict of {name: DataSpec}

        Returns:
            Dict of {name: DataSource}
        """
        from clyptq.data.sources.parquet import ParquetSource
        from clyptq.data.spec import OHLCVSpec

        sources = {}

        for spec_name, spec in specs.items():
            if isinstance(spec, OHLCVSpec):
                # Auto-resolve path from spec
                exchange = spec.exchange
                market_type = spec.market_type
                timeframe = spec.timeframe

                # Build path: {PROJECT_ROOT}/data/{market_type}/{exchange}/{timeframe}/
                import clyptq
                project_root = Path(clyptq.__file__).parent.parent
                resolved_path = project_root / "data" / market_type / exchange / timeframe

                sources[spec_name] = ParquetSource(
                    path=resolved_path,
                    timeframe=timeframe,
                )

        return sources

    def _validate_timeframes(self) -> None:
        """Validate that all data timeframes >= OHLCV timeframe."""
        for source_name, source in self.sources.items():
            source_minutes = timeframe_to_minutes(source.timeframe)
            if source_minutes < self._system_clock_minutes:
                raise ValueError(
                    f"Source '{source_name}' timeframe ({source.timeframe}) "
                    f"is smaller than OHLCV timeframe ({self.system_clock}). "
                    f"All data timeframes must be >= OHLCV timeframe."
                )

    def _validate_ohlcv_available(self) -> None:
        """Validate that OHLCV fields are available."""
        available_fields: set = set()
        for source in self.sources.values():
            available_fields.update(source.fields)

        if "close" not in available_fields:
            raise ValueError(
                "DataProvider requires at least 'close' field. "
                "Add a source that provides OHLCV data."
            )

    def load(
        self,
        symbols: Optional[List[str]] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> "DataProvider":
        """Load data from sources.

        OHLCV source is loaded first to establish system clock timestamps.
        Other sources are then aligned to system clock via forward-fill.

        Args:
            symbols: Symbols to load (optional in research mode - auto-discovers all)
            start: Start date
            end: End date

        Returns:
            self for method chaining
        """
        # Research mode: auto-discover all symbols if not provided
        if symbols is None and self.mode == "research":
            ohlcv_source = self.sources.get("ohlcv")
            if ohlcv_source and hasattr(ohlcv_source, "available_symbols"):
                symbols = ohlcv_source.available_symbols()

        if symbols is None:
            raise ValueError("symbols must be provided (or use mode='research' for auto-discovery)")

        self._symbols = symbols

        # Load OHLCV first to establish system clock timestamps
        ohlcv_source_name = "ohlcv"
        if ohlcv_source_name not in self.sources:
            # Find source with close field
            for name, source in self.sources.items():
                if "close" in source.fields:
                    ohlcv_source_name = name
                    break

        # Load OHLCV source first
        if ohlcv_source_name in self.sources:
            ohlcv_source = self.sources[ohlcv_source_name]
            ohlcv_data = ohlcv_source.load(symbols=symbols, start=start, end=end)
            self._source_fields[ohlcv_source_name] = []

            # Cache OHLCV data (no alignment needed, this IS the system clock)
            for field, df in ohlcv_data.items():
                if df.empty:
                    continue
                prefixed_key = f"{ohlcv_source_name}:{field}"
                self._data[prefixed_key] = df
                self._source_fields[ohlcv_source_name].append(field)
                if field not in self._data:
                    self._data[field] = df

            # Build timestamp index from OHLCV close
            if "close" in self._data:
                self._timestamps = self._data["close"].index.tolist()

        # Load other sources and align to system clock
        for source_name, source in self.sources.items():
            if source_name == ohlcv_source_name:
                continue  # Already loaded

            source_data = source.load(symbols=symbols, start=start, end=end)
            self._source_fields[source_name] = []
            self._cache_data(source_name, source_data)

        # Fallback: build timestamps from any available data
        if not self._timestamps and self._data:
            ref_df = next(iter(self._data.values()))
            self._timestamps = ref_df.index.tolist()

        # Initialize clock for backtest mode
        if self._timestamps and isinstance(self.clock, BacktestClock):
            self.clock.reset(self._timestamps[0], self._timestamps[-1])

        self._current_idx = -1
        self._current_ts = None
        self._last_rebalance_ts = None

        # Mark as loaded BEFORE computing in_universe (since it needs to access data)
        self._loaded = True

        # Initialize in_universe
        # Research mode: compute vectorized in_universe mask (T x N) using Universe
        if self.mode == "research" and self.universe is not None:
            try:
                # Universe.compute_in_universe returns DataFrame (T x N)
                self._in_universe_mask = self.universe.compute_in_universe(self)
                # For backward compat, also set _in_universe as last row
                if self._in_universe_mask is not None and not self._in_universe_mask.empty:
                    self._in_universe = self._in_universe_mask.iloc[-1]
                else:
                    self._in_universe = pd.Series({s: True for s in self._symbols})
            except Exception as e:
                # Fallback: all symbols in universe
                import warnings
                warnings.warn(f"Universe filter failed: {e}. Using all symbols.")
                self._in_universe = pd.Series({s: True for s in self._symbols})
                self._in_universe_mask = None
        else:
            self._in_universe = pd.Series({s: True for s in self._symbols})
            self._in_universe_mask = None

        return self

    # --- Factory Methods ---

    @classmethod
    def from_parquet(
        cls,
        path: Union[str, Path],
        symbols: List[str],
        rebalance_freq: str = "1d",
        timeframe: str = "1d",
        universe: Optional["BaseUniverse"] = None,
    ) -> "DataProvider":
        """Create DataProvider from Parquet files.

        Convenient factory method for the most common use case.

        Args:
            path: Path to directory (symbol-per-file) or single parquet file
            symbols: Symbols to load
            rebalance_freq: Rebalancing frequency ("1h", "1d", "1w")
            timeframe: Data timeframe
            universe: Optional universe (defaults to StaticUniverse)

        Returns:
            DataProvider (not yet loaded - call .load() to load data)

        Example:
            ```python
            provider = DataProvider.from_parquet(
                path="data/crypto/",
                symbols=["BTC", "ETH"],
                rebalance_freq="1d",
            )
            provider.load(start=start, end=end)
            ```
        """
        from clyptq.data.sources.parquet import ParquetSource
        from clyptq.universe import StaticUniverse

        source = ParquetSource(path=path, timeframe=timeframe)

        if universe is None:
            universe = StaticUniverse(symbols=symbols)

        provider = cls(
            universe=universe,
            sources={"ohlcv": source},
            rebalance_freq=rebalance_freq,
            mode="backtest",
        )

        # Store symbols for later load()
        provider._symbols = symbols

        return provider

    @classmethod
    def from_memory(
        cls,
        data: Dict[str, pd.DataFrame],
        symbols: List[str],
        rebalance_freq: str = "1d",
        timeframe: str = "1d",
        universe: Optional["BaseUniverse"] = None,
    ) -> "DataProvider":
        """Create DataProvider from in-memory DataFrames.

        Useful for testing or when data is already loaded.

        Args:
            data: Dict of {field: DataFrame (T x N)}
            symbols: Symbols list
            rebalance_freq: Rebalancing frequency
            timeframe: Data timeframe
            universe: Optional universe

        Returns:
            DataProvider (not yet loaded - call .load() to load data)

        Example:
            ```python
            provider = DataProvider.from_memory(
                data={"close": close_df, "volume": volume_df},
                symbols=["BTC", "ETH"],
            )
            provider.load()
            ```
        """
        from clyptq.data.sources.memory import MemorySource
        from clyptq.universe import StaticUniverse

        source = MemorySource(data=data, timeframe=timeframe)

        if universe is None:
            universe = StaticUniverse(symbols=symbols)

        provider = cls(
            universe=universe,
            sources={"ohlcv": source},
            rebalance_freq=rebalance_freq,
            mode="backtest",
        )

        provider._symbols = symbols

        return provider

    @classmethod
    def from_live(
        cls,
        collector: "DataCollector",
        symbols: List[str],
        rebalance_freq: str = "1d",
        timeframe: str = "1m",
        max_bars: int = 500,
        universe: Optional["BaseUniverse"] = None,
    ) -> "DataProvider":
        """Create DataProvider for live trading.

        Args:
            collector: Data collector (CCXTCollector, etc.)
            symbols: Symbols to stream
            rebalance_freq: Rebalancing frequency
            timeframe: Data timeframe
            max_bars: Buffer size limit
            universe: Optional universe

        Returns:
            DataProvider (call .start_live() to begin streaming)

        Example:
            ```python
            collector = CCXTCollector(exchange="binance")
            provider = DataProvider.from_live(
                collector=collector,
                symbols=["BTC", "ETH"],
                rebalance_freq="1h",
            )
            provider.start_live()
            ```
        """
        from clyptq.data.sources.live import LiveSource
        from clyptq.universe import StaticUniverse

        source = LiveSource(collector=collector, timeframe=timeframe)

        if universe is None:
            universe = StaticUniverse(symbols=symbols)

        provider = cls(
            universe=universe,
            sources={"ohlcv": source},
            rebalance_freq=rebalance_freq,
            mode="live",
            max_bars=max_bars,
        )

        provider._symbols = symbols

        return provider

    def _cache_data(
        self,
        source_name: str,
        data: Dict[str, pd.DataFrame],
    ) -> None:
        """Cache data from source.

        If source timeframe > system clock, align data via forward-fill.
        """
        source = self.sources[source_name]
        source_minutes = timeframe_to_minutes(source.timeframe)

        for field, df in data.items():
            if df.empty:
                continue

            # Align larger timeframe data to system clock via ffill
            if source_minutes > self._system_clock_minutes:
                df = self._align_to_system_clock(df)

            # Store with source prefix
            prefixed_key = f"{source_name}:{field}"
            self._data[prefixed_key] = df
            self._source_fields[source_name].append(field)

            # Also store without prefix (first source wins)
            if field not in self._data:
                self._data[field] = df

    def _align_to_system_clock(self, df: pd.DataFrame) -> pd.DataFrame:
        """Align larger timeframe data to system clock via forward-fill.

        Args:
            df: DataFrame at larger timeframe

        Returns:
            DataFrame aligned to system clock timestamps
        """
        if not self._timestamps:
            # No reference timestamps yet, just return as-is
            return df

        # Create system clock index
        system_index = pd.DatetimeIndex(self._timestamps)

        # Reindex and forward-fill
        aligned = df.reindex(system_index, method="ffill")

        return aligned

    def tick(self) -> bool:
        """Advance clock by one system_clock interval.

        Returns:
            True if more data available, False if finished
        """
        if not self._loaded:
            raise RuntimeError("DataProvider not loaded. Call load() first.")

        self._current_idx += 1

        if self._current_idx >= len(self._timestamps):
            return False

        self._current_ts = self._timestamps[self._current_idx]
        return True

    def _update_in_universe(self) -> None:
        """Update in_universe membership for current timestamp.

        Uses Universe.compute_in_universe() for proper filter/scoring logic.
        Timeline logging enabled for debugging (waveform-style).
        """
        from clyptq.infra.utils import get_logger

        logger = get_logger(__name__)

        if self._current_ts is None:
            self._in_universe = pd.Series({s: True for s in self._symbols})
            logger.debug(
                "[Timeline] in_universe: all symbols (no timestamp)",
                extra={"timestamp": None, "count": len(self._symbols)},
            )
            return

        # Get available symbols at current timestamp (non-NaN close)
        available = set()
        if "close" in self._data:
            close_data = self._data["close"].loc[:self._current_ts]
            if not close_data.empty:
                last_row = close_data.iloc[-1]
                available = set(last_row.dropna().index)

        logger.debug(
            f"[Timeline] {self._current_ts.isoformat()} | Available symbols (N): {len(available)}",
            extra={"timestamp": self._current_ts.isoformat(), "n_available": len(available)},
        )

        # Apply universe filters if available
        try:
            if self.universe is not None:
                # New Universe system: compute_in_universe returns boolean mask
                if hasattr(self.universe, 'compute_in_universe'):
                    in_universe_mask = self.universe.compute_in_universe(self)

                    if in_universe_mask is not None and not in_universe_mask.empty:
                        # Get mask at current timestamp
                        if self._current_ts in in_universe_mask.index:
                            current_mask = in_universe_mask.loc[self._current_ts]
                        else:
                            # Find closest timestamp <= current
                            valid_ts = in_universe_mask.index[in_universe_mask.index <= self._current_ts]
                            if len(valid_ts) > 0:
                                current_mask = in_universe_mask.loc[valid_ts[-1]]
                            else:
                                current_mask = pd.Series({s: s in available for s in self._symbols})

                        self._in_universe = current_mask.reindex(self._symbols, fill_value=False)

                        n_tradeable = len(available)
                        n_in_universe = self._in_universe.sum()
                        logger.debug(
                            f"[Timeline] {self._current_ts.isoformat()} | "
                            f"Tradeable (M): {n_tradeable} → In Universe (n): {n_in_universe}",
                            extra={
                                "timestamp": self._current_ts.isoformat(),
                                "n_tradeable": n_tradeable,
                                "n_in_universe": int(n_in_universe),
                            },
                        )
                        return

                # Legacy: get_symbols method
                elif hasattr(self.universe, 'get_symbols'):
                    class _MinimalStore:
                        def __init__(self, syms):
                            self._syms = syms
                        def symbols(self):
                            return self._syms

                    filtered = self.universe.get_symbols(
                        timestamp=self._current_ts,
                        data_store=_MinimalStore(list(available)),
                    )
                    available = set(filtered)

        except Exception as e:
            logger.warning(
                f"[Timeline] Universe filter error: {e}",
                extra={"timestamp": self._current_ts.isoformat(), "error": str(e)},
            )

        # Fallback: use available symbols
        self._in_universe = pd.Series({
            s: s in available
            for s in self._symbols
        })

        logger.debug(
            f"[Timeline] {self._current_ts.isoformat()} | In Universe: {self._in_universe.sum()}",
            extra={"timestamp": self._current_ts.isoformat(), "n_in_universe": int(self._in_universe.sum())},
        )

    def should_rebalance(self) -> bool:
        """Check if current timestamp is a rebalancing point."""
        if self._current_ts is None:
            return False

        if self._last_rebalance_ts is None:
            return True

        delta_minutes = (self._current_ts - self._last_rebalance_ts).total_seconds() / 60
        return delta_minutes >= self._rebalance_minutes

    def mark_rebalanced(self) -> None:
        """Mark current timestamp as rebalanced."""
        self._last_rebalance_ts = self._current_ts
        self._update_in_universe()

    def __getitem__(self, key: Union[str, tuple]) -> pd.DataFrame:
        """Get data up to current timestamp.

        Supports multi-timeframe access:
            provider["close"]        # System clock timeframe
            provider["close", "4h"]  # Resampled to 4h

        Args:
            key: Field name (str) or tuple of (field, timeframe)

        Returns:
            DataFrame with data up to current timestamp
        """
        if not self._loaded:
            raise RuntimeError("DataProvider not loaded. Call load() first.")

        # Parse key
        if isinstance(key, tuple):
            if len(key) != 2:
                raise KeyError(f"Expected (field, timeframe) tuple, got: {key}")
            field, target_tf = key
        else:
            field = key
            target_tf = None

        if field not in self._data:
            raise KeyError(f"Unknown field: '{field}'")

        # Get raw data
        data = self._data[field]

        # Prevent look-ahead
        if self._current_ts is not None:
            data = data.loc[:self._current_ts]

        # Resample if different timeframe requested
        if target_tf is not None:
            data = self._resample_data(field, data, target_tf)

        # Apply buffer limit
        if self.max_bars is not None and len(data) > self.max_bars:
            data = data.iloc[-self.max_bars:]

        return data

    def _resample_data(
        self,
        field: str,
        data: pd.DataFrame,
        target_tf: str,
    ) -> pd.DataFrame:
        """Resample data to target timeframe.

        Args:
            field: Field name (for aggregation rule selection)
            data: Source data at system clock timeframe
            target_tf: Target timeframe

        Returns:
            Resampled DataFrame
        """
        target_minutes = timeframe_to_minutes(target_tf)

        if target_minutes < self._system_clock_minutes:
            raise ValueError(
                f"Cannot resample to smaller timeframe. "
                f"Target {target_tf} < system clock {self.system_clock}"
            )

        if target_minutes == self._system_clock_minutes:
            return data

        # Check cache
        cache_key = f"{field}:{target_tf}"
        if cache_key in self._resampled_cache:
            cached = self._resampled_cache[cache_key]
            if self._current_ts is not None:
                return cached.loc[:self._current_ts]
            return cached

        # Convert timeframe to pandas offset
        offset = self._timeframe_to_offset(target_tf)

        # Determine aggregation rule
        if field in self._OHLCV_AGG_RULES:
            agg_rule = self._OHLCV_AGG_RULES[field]
        else:
            # Default: last value for non-OHLCV fields
            agg_rule = "last"

        # Resample each column
        resampled = data.resample(offset, label="right", closed="right").agg(agg_rule)

        # Align back to system clock index (forward fill)
        # So 3d data is available at 1d granularity
        aligned = resampled.reindex(data.index, method="ffill")

        # Cache result (aligned version)
        self._resampled_cache[cache_key] = aligned

        return aligned

    def _timeframe_to_offset(self, tf: str) -> str:
        """Convert timeframe string to pandas offset string."""
        minutes = timeframe_to_minutes(tf)

        if minutes >= 10080 and minutes % 10080 == 0:
            return f"{minutes // 10080}W"
        elif minutes >= 1440 and minutes % 1440 == 0:
            return f"{minutes // 1440}D"
        elif minutes >= 60 and minutes % 60 == 0:
            return f"{minutes // 60}h"
        else:
            return f"{minutes}min"

    def get(
        self,
        key: Union[str, tuple],
        default: Any = None,
    ) -> Optional[pd.DataFrame]:
        """Get data with default.

        Args:
            key: Field name (str) or tuple of (field, timeframe)
            default: Default value if field not found

        Returns:
            DataFrame or default value
        """
        try:
            return self[key]
        except KeyError:
            return default

    @property
    def current_timestamp(self) -> Optional[datetime]:
        """Get current timestamp."""
        return self._current_ts

    @property
    def in_universe(self) -> pd.Series:
        """Get current in_universe membership (last row of mask)."""
        return self._in_universe

    @property
    def in_universe_mask(self) -> Optional[pd.DataFrame]:
        """Get full in_universe mask (T x N DataFrame).

        Only available in research mode after load().
        Each row is a timestamp, each column is a symbol.
        True = symbol is in universe at that timestamp.

        Returns:
            DataFrame (T x N) or None if not computed
        """
        return self._in_universe_mask

    @property
    def symbols(self) -> List[str]:
        """Get list of all symbols (N)."""
        return self._symbols

    @property
    def universe_symbols(self) -> List[str]:
        """Get symbols currently in universe (n) at last timestamp."""
        return [s for s, in_univ in self._in_universe.items() if in_univ]

    @property
    def n_universe(self) -> int:
        """Get number of symbols in universe (n) at last timestamp."""
        return int(self._in_universe.sum())

    @property
    def start(self) -> Optional[datetime]:
        """Get start timestamp of loaded data."""
        if self._timestamps:
            return self._timestamps[0]
        return None

    @property
    def end(self) -> Optional[datetime]:
        """Get end timestamp of loaded data."""
        if self._timestamps:
            return self._timestamps[-1]
        return None

    @property
    def fields(self) -> List[str]:
        """Get list of available fields."""
        return list(self._field_to_source.keys())

    @property
    def loaded_fields(self) -> List[str]:
        """Get list of loaded fields."""
        return list(self._data.keys())

    def source_view(self, source_name: str) -> "SourceView":
        """Get a view of data for a specific source."""
        if source_name not in self.sources:
            raise KeyError(f"Unknown source: '{source_name}'")
        return SourceView(self, source_name)

    def current_prices(self) -> Dict[str, float]:
        """Get current prices for all symbols."""
        if "close" in self._data and self._current_ts is not None:
            close_data = self._data["close"].loc[:self._current_ts]
            if not close_data.empty:
                return close_data.iloc[-1].dropna().to_dict()
        return {}

    # --- Strata Methods (Group/Category Metadata) ---

    def get_strata(self, name: str) -> Dict[str, str]:
        """Get strata (group/category) mapping for symbols.

        Strata provide categorical metadata for symbols, enabling group-based
        operations like sector neutralization, stratified ranking, etc.

        Args:
            name: Strata name (e.g., "layer", "market_cap_tier", "volatility_tier")

        Returns:
            Dict mapping symbol -> category (e.g., {"BTC": "L1", "UNI": "DeFi"})

        Available strata (mock data - TODO: replace with real data):
            - "layer": Blockchain layer classification (L1, L2, DeFi, Meme, etc.)
            - "market_cap_tier": Market cap tier (mega, large, mid, small, micro)
            - "volatility_tier": Volatility tier (low, medium, high, extreme)

        Example:
            >>> provider.get_strata("layer")
            {"BTC": "L1", "ETH": "L1", "ARB": "L2", "UNI": "DeFi", ...}

            >>> # Use with by_* operators
            >>> from clyptq.operator import by_demean
            >>> neutralized = by_demean(alpha, strata=provider.get_strata("layer"))
        """
        # TODO: Replace with real strata loading from DB/API/file
        # TODO: Support time-varying strata (symbol category can change over time)
        if name not in self._MOCK_CRYPTO_STRATA:
            available = list(self._MOCK_CRYPTO_STRATA.keys())
            raise KeyError(
                f"Unknown strata: '{name}'. Available: {available}"
            )
        return self._MOCK_CRYPTO_STRATA[name].copy()

    def list_strata(self) -> List[str]:
        """List available strata names.

        Returns:
            List of strata names

        Example:
            >>> provider.list_strata()
            ['layer', 'market_cap_tier', 'volatility_tier']
        """
        # TODO: Replace with real strata listing
        return list(self._MOCK_CRYPTO_STRATA.keys())

    def get_strata_values(self, name: str) -> List[str]:
        """Get unique values/categories for a strata.

        Args:
            name: Strata name

        Returns:
            List of unique category values

        Example:
            >>> provider.get_strata_values("layer")
            ['L1', 'L2', 'DeFi', 'Meme', 'Exchange', 'Metaverse', 'Infrastructure', 'Payment']
        """
        strata = self.get_strata(name)
        return list(set(strata.values()))

    def get_symbols_by_strata(self, name: str, value: str) -> List[str]:
        """Get symbols belonging to a specific strata category.

        Args:
            name: Strata name
            value: Category value

        Returns:
            List of symbols in that category

        Example:
            >>> provider.get_symbols_by_strata("layer", "DeFi")
            ['UNI', 'AAVE', 'LINK', 'MKR', 'CRV', 'SNX', 'COMP', 'SUSHI', 'YFI', '1INCH']
        """
        strata = self.get_strata(name)
        return [symbol for symbol, cat in strata.items() if cat == value]

    # --- Live Mode Methods ---

    def start_live(self, symbols: List[str]) -> None:
        """Start live data streaming."""
        if self.mode != "live":
            raise RuntimeError("start_live() only available in live mode")

        self._symbols = symbols
        self._in_universe = pd.Series({s: True for s in self._symbols})

        # Subscribe to each source that supports it
        for source_name, source in self.sources.items():
            try:
                source.subscribe(
                    symbols=symbols,
                    on_data=lambda sym, data: self._on_live_data(source_name, sym, data),
                )
            except NotImplementedError:
                pass

        self._loaded = True

    def _on_live_data(self, source_name: str, symbol: str, data: Dict[str, Any]) -> None:
        """Handle incoming live data."""
        timestamp = data.get("timestamp", datetime.utcnow())
        if isinstance(timestamp, (int, float)):
            timestamp = datetime.fromtimestamp(timestamp / 1000)

        self._current_ts = timestamp

        # Update data buffer
        for field in ["open", "high", "low", "close", "volume"]:
            if field not in data:
                continue

            key = f"{source_name}:{field}"
            if key not in self._data:
                self._data[key] = pd.DataFrame(columns=self._symbols)
                if field not in self._data:
                    self._data[field] = self._data[key]

            # Add new row
            if timestamp not in self._data[key].index:
                new_row = pd.DataFrame(
                    index=[timestamp],
                    columns=self._data[key].columns,
                )
                self._data[key] = pd.concat([self._data[key], new_row])

            self._data[key].loc[timestamp, symbol] = data[field]

            # Trim buffer
            if self.max_bars and len(self._data[key]) > self.max_bars:
                self._data[key] = self._data[key].iloc[-self.max_bars:]

    def stop_live(self) -> None:
        """Stop live data streaming."""
        for source in self.sources.values():
            if hasattr(source, '_subscription') and source._subscription:
                source._subscription.unsubscribe()

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        ts = self._current_ts.isoformat() if self._current_ts else "None"
        return (
            f"DataProvider(system_clock={self.system_clock}, "
            f"rebalance={self.rebalance_freq}, ts={ts}, {status})"
        )


class SourceView:
    """View of DataProvider for a specific data source."""

    def __init__(self, provider: DataProvider, source_name: str):
        self._provider = provider
        self._source_name = source_name

    def __getitem__(self, field: str) -> pd.DataFrame:
        """Get field data from this source."""
        prefixed_key = f"{self._source_name}:{field}"

        if prefixed_key not in self._provider._data:
            available = self._provider._source_fields.get(self._source_name, [])
            raise KeyError(
                f"Field '{field}' not found in source '{self._source_name}'. "
                f"Available: {available}"
            )

        data = self._provider._data[prefixed_key]

        if self._provider._current_ts is not None:
            data = data.loc[:self._provider._current_ts]

        if self._provider.max_bars is not None and len(data) > self._provider.max_bars:
            data = data.iloc[-self._provider.max_bars:]

        return data

    def get(self, field: str, default: Any = None) -> Optional[pd.DataFrame]:
        """Get field data with default."""
        try:
            return self[field]
        except KeyError:
            return default

    @property
    def fields(self) -> List[str]:
        """Get available fields for this source."""
        return self._provider._source_fields.get(self._source_name, [])

    @property
    def current_timestamp(self) -> Optional[datetime]:
        """Get current timestamp."""
        return self._provider.current_timestamp

    @property
    def in_universe(self) -> pd.Series:
        """Get current in_universe membership."""
        return self._provider.in_universe

    @property
    def symbols(self) -> List[str]:
        """Get list of all symbols."""
        return self._provider.symbols

    def __repr__(self) -> str:
        return f"SourceView(source={self._source_name}, fields={self.fields})"
