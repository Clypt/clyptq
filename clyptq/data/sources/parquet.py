"""Parquet file DataSource.

ParquetSource loads data from Parquet files.

Directory structure (exchange-based):
    data/
    └── {exchange}/
        └── {market_type}/
            └── {timeframe}/
                ├── BTCUSDT.parquet
                └── ETHUSDT.parquet

Example:
    data/binance/spot/1d/BTCUSDT.parquet
    data/binance/futures/1h/ETHUSDT.parquet

Also supports legacy layouts:
1. Direct path to directory with parquet files
2. Single file with symbol column
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import pandas as pd

from clyptq.data.sources.base import DataSource, OHLCVFields


def _read_parquet_safe(path: Path) -> pd.DataFrame:
    """Read parquet with fallback for PyArrow version issues."""
    import pyarrow.parquet as pq
    # PyArrow 21.0.0 has bugs with default settings, use legacy dataset
    try:
        table = pq.read_table(str(path), use_legacy_dataset=True)
        return table.to_pandas()
    except Exception:
        # Fallback to pandas default
        return pd.read_parquet(path)


# Supported exchanges
Exchange = Literal["binance", "okx", "bybit", "upbit"]
MarketType = Literal["spot", "futures", "margin"]

# Default data root
DEFAULT_DATA_ROOT = Path("data")


class ParquetSource(DataSource):
    """Parquet file data source.

    Loads OHLCV data from Parquet files organized by exchange.

    Example:
        ```python
        # Exchange-based (recommended)
        source = ParquetSource(
            exchange="binance",
            market_type="spot",
            timeframe="1d",
        )
        data = source.load(symbols=["BTCUSDT", "ETHUSDT"])

        # Direct path (legacy)
        source = ParquetSource(path="data/custom/")
        data = source.load(symbols=["BTCUSDT"])
        ```
    """

    def __init__(
        self,
        exchange: Optional[Exchange] = None,
        market_type: MarketType = "spot",
        timeframe: str = "1d",
        path: Optional[Union[str, Path]] = None,
        data_root: Optional[Union[str, Path]] = None,
        name: Optional[str] = None,
        timestamp_col: str = "timestamp",
        symbol_col: str = "symbol",
    ):
        """Initialize ParquetSource.

        Args:
            exchange: Exchange name (binance, okx, bybit, upbit)
            market_type: Market type (spot, futures, margin)
            timeframe: Data timeframe (1m, 5m, 1h, 4h, 1d, etc.)
            path: Direct path to data (overrides exchange-based path)
            data_root: Root directory for data (default: "data/")
            name: Source name
            timestamp_col: Name of timestamp column
            symbol_col: Name of symbol column (for single-file layout)
        """
        super().__init__(name=name or f"{exchange}_{market_type}", timeframe=timeframe)

        self.exchange = exchange
        self.market_type = market_type
        self.timestamp_col = timestamp_col
        self.symbol_col = symbol_col

        # Determine path
        if path is not None:
            self.path = Path(path)
        elif exchange is not None:
            root = Path(data_root) if data_root else DEFAULT_DATA_ROOT
            self.path = root / exchange / market_type / timeframe
        else:
            raise ValueError("Either 'exchange' or 'path' must be provided")

        # Determine layout
        self._is_single_file = self.path.is_file() if self.path.exists() else False
        self._cached_data: Optional[pd.DataFrame] = None

    @property
    def fields(self) -> List[str]:
        """OHLCV fields."""
        return OHLCVFields.ALL

    @classmethod
    def for_exchange(
        cls,
        exchange: Exchange,
        market_type: MarketType = "spot",
        timeframe: str = "1d",
        data_root: Optional[Union[str, Path]] = None,
    ) -> "ParquetSource":
        """Factory method for exchange-based source.

        Args:
            exchange: Exchange name
            market_type: Market type
            timeframe: Data timeframe
            data_root: Root directory

        Returns:
            Configured ParquetSource
        """
        return cls(
            exchange=exchange,
            market_type=market_type,
            timeframe=timeframe,
            data_root=data_root,
        )

    def load(
        self,
        symbols: List[str],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        timeframe: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Load data for symbols from Parquet files.

        Args:
            symbols: Symbols to load (e.g., ["BTCUSDT", "ETHUSDT"])
            start: Start datetime
            end: End datetime
            timeframe: Target timeframe (resampling if needed)

        Returns:
            Dict[field, DataFrame (T x N)]
        """
        target_tf = timeframe or self.timeframe
        needs_resample = target_tf != self.timeframe

        if not self.path.exists():
            raise FileNotFoundError(
                f"Data path not found: {self.path}\n"
                f"Expected structure: data/{self.exchange}/{self.market_type}/{self.timeframe}/"
            )

        if self._is_single_file:
            return self._load_single_file(symbols, start, end, target_tf, needs_resample)
        else:
            return self._load_multi_file(symbols, start, end, target_tf, needs_resample)

    def _load_single_file(
        self,
        symbols: List[str],
        start: Optional[datetime],
        end: Optional[datetime],
        target_tf: str,
        needs_resample: bool,
    ) -> Dict[str, pd.DataFrame]:
        """Load from single parquet file with symbol column."""
        if self._cached_data is None:
            self._cached_data = _read_parquet_safe(self.path)

        df = self._cached_data.copy()

        # Filter by symbols
        if self.symbol_col in df.columns:
            df = df[df[self.symbol_col].isin(symbols)]

        # Handle timestamp
        if self.timestamp_col in df.columns:
            df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])
            df = df.set_index(self.timestamp_col)

        # Apply date filters
        if start is not None:
            df = df[df.index >= start]
        if end is not None:
            df = df[df.index <= end]

        # Pivot to wide format
        result: Dict[str, pd.DataFrame] = {}
        for field in self.fields:
            if field in df.columns:
                wide = df.pivot_table(
                    index=df.index,
                    columns=self.symbol_col,
                    values=field,
                    aggfunc="last",
                )
                if needs_resample:
                    wide = self._resample_wide(wide, field, target_tf)
                result[field] = wide

        return result

    def _load_multi_file(
        self,
        symbols: List[str],
        start: Optional[datetime],
        end: Optional[datetime],
        target_tf: str,
        needs_resample: bool,
    ) -> Dict[str, pd.DataFrame]:
        """Load from multiple parquet files (one per symbol)."""
        result: Dict[str, pd.DataFrame] = {}

        for field in self.fields:
            series_dict = {}

            for symbol in symbols:
                file_path = self.path / f"{symbol}.parquet"
                if not file_path.exists():
                    continue

                df = _read_parquet_safe(file_path)

                # Handle timestamp
                if self.timestamp_col in df.columns:
                    df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])
                    df = df.set_index(self.timestamp_col)

                # Apply date filters
                if start is not None:
                    df = df[df.index >= start]
                if end is not None:
                    df = df[df.index <= end]

                if len(df) == 0:
                    continue

                # Resample if needed
                if needs_resample:
                    df = self._resample(df, target_tf)

                if field in df.columns and len(df) > 0:
                    series_dict[symbol] = df[field]

            if series_dict:
                result[field] = pd.DataFrame(series_dict)

        return result

    def _resample(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample OHLCV data to target timeframe."""
        rule = self._timeframe_to_rule(timeframe)

        agg_dict = {}
        if "open" in df.columns:
            agg_dict["open"] = "first"
        if "high" in df.columns:
            agg_dict["high"] = "max"
        if "low" in df.columns:
            agg_dict["low"] = "min"
        if "close" in df.columns:
            agg_dict["close"] = "last"
        if "volume" in df.columns:
            agg_dict["volume"] = "sum"

        return df.resample(rule).agg(agg_dict).dropna()

    def _resample_wide(
        self,
        df: pd.DataFrame,
        field: str,
        timeframe: str,
    ) -> pd.DataFrame:
        """Resample wide-format DataFrame."""
        rule = self._timeframe_to_rule(timeframe)

        if field == "open":
            return df.resample(rule).first().dropna()
        elif field == "high":
            return df.resample(rule).max().dropna()
        elif field == "low":
            return df.resample(rule).min().dropna()
        elif field == "close":
            return df.resample(rule).last().dropna()
        elif field == "volume":
            return df.resample(rule).sum().dropna()
        else:
            return df.resample(rule).last().dropna()

    def _timeframe_to_rule(self, timeframe: str) -> str:
        """Convert timeframe to pandas resample rule."""
        if timeframe.endswith("m"):
            return f"{timeframe[:-1]}min"
        elif timeframe.endswith("h"):
            return f"{timeframe[:-1]}h"
        elif timeframe.endswith("d"):
            return f"{timeframe[:-1]}D"
        elif timeframe.endswith("w"):
            return f"{timeframe[:-1]}W"
        else:
            raise ValueError(f"Unknown timeframe: {timeframe}")

    def available_symbols(self) -> List[str]:
        """Get list of available symbols."""
        if not self.path.exists():
            return []

        if self._is_single_file:
            if self._cached_data is None:
                self._cached_data = _read_parquet_safe(self.path)
            if self.symbol_col in self._cached_data.columns:
                return sorted(self._cached_data[self.symbol_col].unique().tolist())
            return []
        else:
            return sorted([f.stem for f in self.path.glob("*.parquet")])

    def __repr__(self) -> str:
        if self.exchange:
            return f"ParquetSource({self.exchange}/{self.market_type}/{self.timeframe})"
        return f"ParquetSource({self.path})"
