"""Timeframe utilities for system clock calculation.

System clock = GCD of all timeframes (signals + rebalance).
This determines the minimum tick interval for backtesting.
"""

from functools import reduce
from math import gcd
from typing import List


# Timeframe to minutes mapping
TIMEFRAME_MINUTES = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "6h": 360,
    "8h": 480,
    "12h": 720,
    "1d": 1440,
    "3d": 4320,
    "1w": 10080,
    "1M": 43200,  # ~30 days
}


def timeframe_to_minutes(tf: str) -> int:
    """Convert timeframe string to minutes.

    Args:
        tf: Timeframe string (e.g., "15m", "1h", "1d")

    Returns:
        Number of minutes

    Examples:
        >>> timeframe_to_minutes("15m")
        15
        >>> timeframe_to_minutes("4h")
        240
        >>> timeframe_to_minutes("1d")
        1440
    """
    if tf in TIMEFRAME_MINUTES:
        return TIMEFRAME_MINUTES[tf]

    # Parse custom format
    unit = tf[-1].lower()
    try:
        value = int(tf[:-1])
    except ValueError:
        raise ValueError(f"Invalid timeframe format: {tf}")

    if unit == "m":
        return value
    elif unit == "h":
        return value * 60
    elif unit == "d":
        return value * 1440
    elif unit == "w":
        return value * 10080
    else:
        raise ValueError(f"Unknown timeframe unit: {unit}")


def minutes_to_timeframe(minutes: int) -> str:
    """Convert minutes to timeframe string.

    Args:
        minutes: Number of minutes

    Returns:
        Timeframe string

    Examples:
        >>> minutes_to_timeframe(15)
        "15m"
        >>> minutes_to_timeframe(240)
        "4h"
        >>> minutes_to_timeframe(1440)
        "1d"
    """
    if minutes >= 10080 and minutes % 10080 == 0:
        return f"{minutes // 10080}w"
    elif minutes >= 1440 and minutes % 1440 == 0:
        return f"{minutes // 1440}d"
    elif minutes >= 60 and minutes % 60 == 0:
        return f"{minutes // 60}h"
    else:
        return f"{minutes}m"


def calculate_system_clock(timeframes: List[str]) -> str:
    """Calculate system clock as GCD of all timeframes.

    System clock determines the tick interval for backtesting.
    Using GCD ensures all timeframes align properly.

    Args:
        timeframes: List of timeframe strings (e.g., ["15m", "1h", "1d"])

    Returns:
        System clock timeframe string

    Examples:
        >>> calculate_system_clock(["15m", "30m", "1h"])
        "15m"
        >>> calculate_system_clock(["4h", "1d"])
        "4h"
        >>> calculate_system_clock(["1h", "1d", "1w"])
        "1h"
    """
    if not timeframes:
        raise ValueError("At least one timeframe required")

    minutes_list = [timeframe_to_minutes(tf) for tf in timeframes]
    gcd_minutes = reduce(gcd, minutes_list)

    return minutes_to_timeframe(gcd_minutes)
