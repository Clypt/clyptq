"""Base classes for ClyptQ."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional

import pandas as pd

from clyptq.core.types import Constraints, Fill, Order

if TYPE_CHECKING:
    from clyptq.data.stores.store import DataView
    from clyptq.data.stores.mtf_store import MultiTimeframeStore


class Factor(ABC):
    """Abstract base class for alpha factors."""

    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def compute(self, data: "DataView") -> Dict[str, float]:
        """Compute factor scores for all available symbols."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class MultiTimeframeFactor(ABC):
    """Base class for factors using multiple timeframes."""

    def __init__(
        self,
        timeframes: List[str],
        lookbacks: Optional[Dict[str, int]] = None,
        name: Optional[str] = None,
    ):
        from clyptq.data.stores.mtf_store import MultiTimeframeStore

        self.timeframes = timeframes
        self.lookbacks = lookbacks or {tf: 20 for tf in timeframes}
        self.name = name or self.__class__.__name__

        for tf in timeframes:
            if tf not in MultiTimeframeStore.VALID_TIMEFRAMES:
                raise ValueError(
                    f"Invalid timeframe '{tf}'. Must be one of {MultiTimeframeStore.VALID_TIMEFRAMES}"
                )

    def compute(
        self,
        mtf_store: "MultiTimeframeStore",
        timestamp: datetime,
        symbols: List[str],
    ) -> Dict[str, float]:
        """Compute factor scores across multiple timeframes."""
        scores = {}

        for symbol in symbols:
            has_data = all(
                mtf_store.has_sufficient_data(
                    symbol, tf, timestamp, self.lookbacks[tf]
                )
                for tf in self.timeframes
            )

            if not has_data:
                continue

            timeframe_data = {}
            for tf in self.timeframes:
                store = mtf_store.get_store(tf)
                data = store._data[symbol]
                available = data[data.index <= timestamp]

                lookback = self.lookbacks[tf]
                window = available.iloc[-lookback:]
                timeframe_data[tf] = window

            score = self.compute_mtf(symbol, timeframe_data, timestamp)
            if score is not None:
                scores[symbol] = score

        return scores

    @abstractmethod
    def compute_mtf(
        self,
        symbol: str,
        timeframe_data: Dict[str, pd.DataFrame],
        timestamp: datetime,
    ) -> Optional[float]:
        """Compute factor score using data from multiple timeframes."""
        pass

    def warmup_periods(self) -> int:
        """Get maximum warmup periods across all timeframes."""
        return max(self.lookbacks.values())

    def required_timeframes(self) -> List[str]:
        """Get list of required timeframes."""
        return self.timeframes.copy()


class Executor(ABC):
    """Base class for order executors."""

    @abstractmethod
    def execute(
        self, orders: List[Order], timestamp: datetime, prices: Dict[str, float]
    ) -> List[Fill]:
        pass


class PortfolioConstructor(ABC):
    """Abstract base class for portfolio construction."""

    @abstractmethod
    def construct(
        self, scores: Dict[str, float], constraints: Constraints
    ) -> Dict[str, float]:
        """Construct target portfolio from factor scores."""
        pass


class Strategy(ABC):
    """Abstract base class for trading strategies."""

    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def factors(self) -> List[Factor]:
        """Get list of alpha factors used by this strategy."""
        pass

    @abstractmethod
    def portfolio_constructor(self) -> PortfolioConstructor:
        """Get portfolio constructor for this strategy."""
        pass

    @abstractmethod
    def constraints(self) -> Constraints:
        """Get portfolio constraints for this strategy."""
        pass

    def schedule(self) -> str:
        """Get rebalancing schedule."""
        return "daily"

    def universe(self) -> Optional[List[str]]:
        """Get trading universe (list of symbols)."""
        return None

    def warmup_periods(self) -> int:
        """Get number of warmup periods required before trading."""
        return 100

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
