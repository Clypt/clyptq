"""Base class for streaming data sources."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Callable, Dict, List


class StreamingDataSource(ABC):
    """Abstract streaming data source."""

    @abstractmethod
    async def start(self, symbols: List[str], on_tick: Callable[[datetime, Dict[str, float]], None]) -> None:
        """Start streaming prices for symbols."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop streaming."""
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """Check if stream is active."""
        pass
