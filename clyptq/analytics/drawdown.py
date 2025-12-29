"""
Drawdown analysis.

Analyze drawdown periods, duration, and recovery patterns.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import numpy as np

from clyptq.core.types import BacktestResult


@dataclass
class DrawdownPeriod:
    """Single drawdown period."""

    start: datetime
    end: datetime
    recovery: Optional[datetime]
    depth: float
    duration_days: int
    recovery_days: Optional[int]

    def to_dict(self) -> dict:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "recovery": self.recovery.isoformat() if self.recovery else None,
            "depth": self.depth,
            "duration_days": self.duration_days,
            "recovery_days": self.recovery_days,
        }


@dataclass
class DrawdownAnalysis:
    """Complete drawdown analysis."""

    max_drawdown: float
    avg_drawdown: float
    drawdown_periods: List[DrawdownPeriod]
    underwater_equity: List[float]
    timestamps: List[datetime]

    def to_dict(self) -> dict:
        return {
            "max_drawdown": self.max_drawdown,
            "avg_drawdown": self.avg_drawdown,
            "drawdown_periods": [d.to_dict() for d in self.drawdown_periods],
            "underwater_equity": self.underwater_equity,
            "timestamps": [ts.isoformat() for ts in self.timestamps],
        }


class DrawdownAnalyzer:
    """
    Analyze drawdown patterns.

    Identifies drawdown periods, calculates duration and depth,
    and provides underwater equity data.
    """

    def __init__(self, min_drawdown: float = 0.01):
        """
        Initialize analyzer.

        Args:
            min_drawdown: Minimum drawdown depth to track (default 1%)
        """
        self.min_drawdown = min_drawdown

    def analyze(self, result: BacktestResult) -> DrawdownAnalysis:
        """
        Analyze drawdowns.

        Args:
            result: Backtest result with snapshots

        Returns:
            DrawdownAnalysis with period details
        """
        if not result.snapshots:
            raise ValueError("No snapshots in result")

        equity = np.array([s.equity for s in result.snapshots])
        timestamps = [s.timestamp for s in result.snapshots]

        underwater = self._calculate_underwater(equity)
        periods = self._find_drawdown_periods(timestamps, underwater)

        drawdowns = [abs(p.depth) for p in periods]
        max_dd = max(drawdowns) if drawdowns else 0.0
        avg_dd = np.mean(drawdowns) if drawdowns else 0.0

        return DrawdownAnalysis(
            max_drawdown=max_dd,
            avg_drawdown=avg_dd,
            drawdown_periods=periods,
            underwater_equity=underwater.tolist(),
            timestamps=timestamps,
        )

    def _calculate_underwater(self, equity: np.ndarray) -> np.ndarray:
        """Calculate underwater (drawdown) values."""
        running_max = np.maximum.accumulate(equity)
        underwater = (equity - running_max) / running_max
        return underwater

    def _find_drawdown_periods(
        self,
        timestamps: List[datetime],
        underwater: np.ndarray,
    ) -> List[DrawdownPeriod]:
        """Find all drawdown periods."""
        periods = []
        in_drawdown = False
        start_idx = 0

        for i in range(len(underwater)):
            dd = underwater[i]

            if not in_drawdown and dd < -self.min_drawdown:
                in_drawdown = True
                start_idx = i

            elif in_drawdown and dd >= 0:
                end_idx = i - 1
                recovery_idx = i

                depth = float(np.min(underwater[start_idx : end_idx + 1]))
                duration = end_idx - start_idx + 1
                recovery_days = recovery_idx - start_idx

                period = DrawdownPeriod(
                    start=timestamps[start_idx],
                    end=timestamps[end_idx],
                    recovery=timestamps[recovery_idx],
                    depth=depth,
                    duration_days=duration,
                    recovery_days=recovery_days,
                )
                periods.append(period)
                in_drawdown = False

        if in_drawdown:
            end_idx = len(underwater) - 1
            depth = float(np.min(underwater[start_idx:]))
            duration = end_idx - start_idx + 1

            period = DrawdownPeriod(
                start=timestamps[start_idx],
                end=timestamps[end_idx],
                recovery=None,
                depth=depth,
                duration_days=duration,
                recovery_days=None,
            )
            periods.append(period)

        return sorted(periods, key=lambda x: abs(x.depth), reverse=True)
