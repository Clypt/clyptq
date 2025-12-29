"""
Data validation utilities for quality checks and cleaning.

Provides functions to validate OHLCV data for common issues:
- Missing values
- Invalid prices (negative, zero, NaN)
- OHLC consistency (high >= low, etc.)
- Outliers and suspicious price movements
- Timestamp gaps and duplicates
"""

from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class ValidationError(Exception):
    """Raised when data validation fails."""

    pass


def validate_ohlcv(df: pd.DataFrame, symbol: str = "unknown") -> None:
    """
    Validate OHLCV DataFrame for common data quality issues.

    Args:
        df: DataFrame with columns [open, high, low, close, volume]
        symbol: Symbol name for error messages

    Raises:
        ValidationError: If validation fails
    """
    # Check required columns
    required = ["open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValidationError(f"{symbol}: Missing columns: {missing}")

    # Check for empty data
    if len(df) == 0:
        raise ValidationError(f"{symbol}: DataFrame is empty")

    # Check for NaN/inf values
    for col in required:
        if df[col].isna().any():
            raise ValidationError(f"{symbol}: NaN values in {col}")
        if np.isinf(df[col]).any():
            raise ValidationError(f"{symbol}: Infinite values in {col}")

    # Check for non-positive prices
    for col in ["open", "high", "low", "close"]:
        if (df[col] <= 0).any():
            raise ValidationError(f"{symbol}: Non-positive prices in {col}")

    # Check for negative volume
    if (df["volume"] < 0).any():
        raise ValidationError(f"{symbol}: Negative volume")

    # Check OHLC consistency
    if (df["high"] < df["low"]).any():
        raise ValidationError(f"{symbol}: High < Low")

    if (df["high"] < df["open"]).any():
        raise ValidationError(f"{symbol}: High < Open")

    if (df["high"] < df["close"]).any():
        raise ValidationError(f"{symbol}: High < Close")

    if (df["low"] > df["open"]).any():
        raise ValidationError(f"{symbol}: Low > Open")

    if (df["low"] > df["close"]).any():
        raise ValidationError(f"{symbol}: Low > Close")

    # Check for duplicate timestamps
    if df.index.duplicated().any():
        raise ValidationError(f"{symbol}: Duplicate timestamps found")


def check_outliers(
    df: pd.DataFrame, symbol: str = "unknown", z_threshold: float = 10.0
) -> List[Tuple[datetime, str]]:
    """
    Check for price outliers using z-score method.

    Args:
        df: DataFrame with OHLCV data
        symbol: Symbol name
        z_threshold: Z-score threshold for outlier detection

    Returns:
        List of (timestamp, reason) tuples for suspected outliers
    """
    outliers = []

    # Calculate returns
    returns = df["close"].pct_change()

    # Z-score of returns
    mean_ret = returns.mean()
    std_ret = returns.std()

    if std_ret > 0:
        z_scores = (returns - mean_ret) / std_ret

        # Find outliers
        mask = np.abs(z_scores) > z_threshold
        for ts in df.index[mask]:
            ret = returns.loc[ts]
            z = z_scores.loc[ts]
            outliers.append((ts, f"Return={ret:.2%}, Z-score={z:.2f}"))

    return outliers


def check_gaps(
    df: pd.DataFrame,
    symbol: str = "unknown",
    expected_freq: str = "1d",
    max_gap_multiplier: float = 2.0,
) -> List[Tuple[datetime, datetime, timedelta]]:
    """
    Check for unexpected gaps in timestamps.

    Args:
        df: DataFrame with OHLCV data
        symbol: Symbol name
        expected_freq: Expected frequency ("1m", "1h", "1d", etc.)
        max_gap_multiplier: Maximum gap as multiple of expected frequency

    Returns:
        List of (start, end, gap_duration) tuples for large gaps
    """
    # Parse expected frequency
    freq_map = {
        "1m": timedelta(minutes=1),
        "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15),
        "1h": timedelta(hours=1),
        "4h": timedelta(hours=4),
        "1d": timedelta(days=1),
    }

    if expected_freq not in freq_map:
        raise ValueError(f"Unknown frequency: {expected_freq}")

    expected_delta = freq_map[expected_freq]
    max_gap = expected_delta * max_gap_multiplier

    # Find gaps
    gaps = []
    timestamps = df.index

    for i in range(1, len(timestamps)):
        gap = timestamps[i] - timestamps[i - 1]
        if gap > max_gap:
            gaps.append((timestamps[i - 1], timestamps[i], gap))

    return gaps


def clean_ohlcv(
    df: pd.DataFrame,
    remove_duplicates: bool = True,
    fill_method: str = "ffill",
    max_fill_limit: int = 3,
) -> pd.DataFrame:
    """
    Clean OHLCV data by handling duplicates and missing values.

    Args:
        df: DataFrame with OHLCV data
        remove_duplicates: Remove duplicate timestamps
        fill_method: Method for filling gaps ('ffill', 'bfill', or None)
        max_fill_limit: Maximum number of consecutive NaNs to fill

    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()

    # Remove duplicates (keep last)
    if remove_duplicates and df_clean.index.duplicated().any():
        df_clean = df_clean[~df_clean.index.duplicated(keep="last")]

    # Fill missing values
    if fill_method:
        if fill_method == "ffill":
            df_clean = df_clean.ffill(limit=max_fill_limit)
        elif fill_method == "bfill":
            df_clean = df_clean.bfill(limit=max_fill_limit)
        else:
            raise ValueError(f"Unknown fill_method: {fill_method}")

    # Drop any remaining NaNs
    df_clean = df_clean.dropna()

    return df_clean


def get_data_quality_report(df: pd.DataFrame, symbol: str = "unknown") -> Dict:
    """
    Generate comprehensive data quality report.

    Args:
        df: DataFrame with OHLCV data
        symbol: Symbol name

    Returns:
        Dictionary with quality metrics
    """
    report = {
        "symbol": symbol,
        "num_bars": len(df),
        "start_date": df.index.min().isoformat(),
        "end_date": df.index.max().isoformat(),
        "duration_days": (df.index.max() - df.index.min()).days,
    }

    # Missing values
    report["missing_values"] = {col: df[col].isna().sum() for col in df.columns}

    # Duplicate timestamps
    report["duplicate_timestamps"] = df.index.duplicated().sum()

    # Price statistics
    report["price_stats"] = {
        "min_close": float(df["close"].min()),
        "max_close": float(df["close"].max()),
        "mean_close": float(df["close"].mean()),
        "std_close": float(df["close"].std()),
    }

    # Volume statistics
    report["volume_stats"] = {
        "min": float(df["volume"].min()),
        "max": float(df["volume"].max()),
        "mean": float(df["volume"].mean()),
        "zero_volume_count": (df["volume"] == 0).sum(),
    }

    # Returns statistics
    returns = df["close"].pct_change().dropna()
    report["returns_stats"] = {
        "mean": float(returns.mean()),
        "std": float(returns.std()),
        "min": float(returns.min()),
        "max": float(returns.max()),
        "skew": float(returns.skew()) if len(returns) > 2 else 0.0,
        "kurtosis": float(returns.kurtosis()) if len(returns) > 3 else 0.0,
    }

    # Outliers
    outliers = check_outliers(df, symbol)
    report["outlier_count"] = len(outliers)

    # Gaps
    gaps = check_gaps(df, symbol)
    report["gap_count"] = len(gaps)

    return report


def assert_valid_ohlcv(df: pd.DataFrame, symbol: str = "unknown") -> None:
    """
    Assert that OHLCV data is valid, raising exception if not.

    This is a convenience function that combines all validation checks.

    Args:
        df: DataFrame with OHLCV data
        symbol: Symbol name

    Raises:
        ValidationError: If any validation fails
    """
    # Basic validation
    validate_ohlcv(df, symbol)

    # Check for too many outliers
    outliers = check_outliers(df, symbol)
    if len(outliers) > len(df) * 0.05:  # More than 5% outliers
        raise ValidationError(
            f"{symbol}: Too many outliers ({len(outliers)}/{len(df)}={len(outliers)/len(df):.1%})"
        )

    # Check for too many gaps
    gaps = check_gaps(df, symbol)
    if len(gaps) > len(df) * 0.1:  # More than 10% gaps
        raise ValidationError(
            f"{symbol}: Too many gaps ({len(gaps)}/{len(df)}={len(gaps)/len(df):.1%})"
        )
