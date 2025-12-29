"""Base executor interface."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List

from clyptq.types import Fill, Order


class Executor(ABC):
    """Base class for order executors."""

    @abstractmethod
    def execute(
        self, orders: List[Order], timestamp: datetime, prices: Dict[str, float]
    ) -> List[Fill]:
        pass
