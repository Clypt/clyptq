"""Clock abstraction for time management.

Clock controls time progression in both backtest and live modes:
- BacktestClock: Tick-based simulation time
- LiveClock: Real-time with async waiting

Design:
- Clock is separate from DataProvider
- Clock determines "current time" for the system
- All components use clock.current_time as reference
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Optional

from clyptq.core.timeframe import (
    calculate_system_clock,
    timeframe_to_minutes,
)


class Clock(ABC):
    """Abstract clock for time management.

    Clock provides:
    - current_time: Current system time
    - tick(): Advance time (backtest) or wait (live)
    - is_aligned(timeframe): Check if current time aligns with timeframe
    """

    def __init__(self, system_clock: str):
        """Initialize clock.

        Args:
            system_clock: Base tick interval (e.g., "15m", "1h")
        """
        self.system_clock = system_clock
        self._tick_minutes = timeframe_to_minutes(system_clock)

    @property
    @abstractmethod
    def current_time(self) -> datetime:
        """Get current time."""
        pass

    @abstractmethod
    def tick(self) -> bool:
        """Advance to next tick.

        Returns:
            True if tick successful, False if end reached
        """
        pass

    @abstractmethod
    def reset(self, start: datetime, end: Optional[datetime] = None) -> None:
        """Reset clock to start time.

        Args:
            start: Start time
            end: End time (optional, for backtest)
        """
        pass

    def is_aligned(self, timeframe: str) -> bool:
        """Check if current time aligns with timeframe boundary.

        Args:
            timeframe: Timeframe to check (e.g., "1h", "4h", "1d")

        Returns:
            True if current time is on timeframe boundary
        """
        tf_minutes = timeframe_to_minutes(timeframe)
        # Minutes since midnight
        minutes_since_midnight = (
            self.current_time.hour * 60 + self.current_time.minute
        )
        return minutes_since_midnight % tf_minutes == 0

    @classmethod
    def from_timeframes(
        cls, timeframes: List[str], **kwargs
    ) -> "Clock":
        """Create clock from list of timeframes.

        System clock = GCD of all timeframes.

        Args:
            timeframes: List of timeframes to support
            **kwargs: Additional arguments for clock

        Returns:
            Clock instance
        """
        system_clock = calculate_system_clock(timeframes)
        return cls(system_clock, **kwargs)


class BacktestClock(Clock):
    """Simulated clock for backtesting.

    Advances time in discrete ticks based on system_clock.
    Tick interval = GCD of all required timeframes.
    """

    def __init__(self, system_clock: str):
        """Initialize backtest clock.

        Args:
            system_clock: Tick interval (e.g., "15m", "1h")
        """
        super().__init__(system_clock)
        self._current_time: Optional[datetime] = None
        self._start_time: Optional[datetime] = None
        self._end_time: Optional[datetime] = None
        self._tick_delta = timedelta(minutes=self._tick_minutes)

    @property
    def current_time(self) -> datetime:
        """Get current simulation time."""
        if self._current_time is None:
            raise RuntimeError("Clock not initialized. Call reset() first.")
        return self._current_time

    def tick(self) -> bool:
        """Advance to next tick.

        Returns:
            True if within time range, False if end reached
        """
        if self._current_time is None:
            raise RuntimeError("Clock not initialized. Call reset() first.")

        next_time = self._current_time + self._tick_delta

        if self._end_time and next_time > self._end_time:
            return False

        self._current_time = next_time
        return True

    def reset(self, start: datetime, end: Optional[datetime] = None) -> None:
        """Reset clock to start time.

        Args:
            start: Start time
            end: End time (backtest boundary)
        """
        self._start_time = start
        self._end_time = end
        self._current_time = start

    def __repr__(self) -> str:
        return (
            f"BacktestClock(system_clock={self.system_clock}, "
            f"current={self._current_time})"
        )


class LiveClock(Clock):
    """Real-time clock for live trading.

    Uses actual system time with async waiting for next tick.
    """

    def __init__(self, system_clock: str):
        """Initialize live clock.

        Args:
            system_clock: Tick interval for alignment
        """
        super().__init__(system_clock)
        self._started = False

    @property
    def current_time(self) -> datetime:
        """Get current real time."""
        return datetime.utcnow()

    def tick(self) -> bool:
        """Wait until next tick boundary.

        In live mode, this should be called with async.
        Synchronous version just returns True immediately.

        Returns:
            Always True (live never ends)
        """
        # For synchronous usage, just return True
        # Actual waiting should be done with async version
        return True

    async def async_tick(self) -> bool:
        """Async wait until next tick boundary.

        Returns:
            Always True (live never ends)
        """
        import asyncio

        now = datetime.utcnow()
        # Calculate next aligned time
        minutes_since_midnight = now.hour * 60 + now.minute
        next_aligned = (
            (minutes_since_midnight // self._tick_minutes + 1)
            * self._tick_minutes
        )

        # Handle day rollover
        if next_aligned >= 1440:
            next_aligned = 0
            next_day = now.date() + timedelta(days=1)
            next_time = datetime.combine(
                next_day, datetime.min.time()
            )
        else:
            next_time = now.replace(
                hour=next_aligned // 60,
                minute=next_aligned % 60,
                second=0,
                microsecond=0,
            )

        # Wait
        wait_seconds = (next_time - now).total_seconds()
        if wait_seconds > 0:
            await asyncio.sleep(wait_seconds)

        return True

    def reset(self, start: datetime, end: Optional[datetime] = None) -> None:
        """Reset clock (no-op for live).

        Args:
            start: Ignored
            end: Ignored
        """
        self._started = True

    def __repr__(self) -> str:
        return f"LiveClock(system_clock={self.system_clock})"
