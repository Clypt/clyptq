"""Base classes for order execution."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List

from clyptq.core.types import Fill, Order


class Executor(ABC):
    """Abstract base class for order executors.

    Executors are responsible for converting orders into fills.
    Different implementations handle backtest (deterministic) vs live (real) execution.

    Responsibilities:
    - Execute orders and return fills
    - Track order status (live)
    - Fetch account balance and positions (live)

    NOT responsible for:
    - Price data fetching (use DataProvider)
    - Order generation (use OrderGenerator)
    """

    @abstractmethod
    def execute(
        self, orders: List[Order], timestamp: datetime, prices: Dict[str, float]
    ) -> List[Fill]:
        """Execute orders and return fills.

        Args:
            orders: List of orders to execute
            timestamp: Execution timestamp
            prices: Current market prices {symbol: price}

        Returns:
            List of fills from executed orders
        """
        pass
