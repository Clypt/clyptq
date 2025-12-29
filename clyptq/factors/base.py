"""
Base class for alpha factor computation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional

from clyptq.data.store import DataView


class Factor(ABC):
    """
    Abstract base class for alpha factors.

    All factors must implement compute() which takes DataView
    and returns {symbol: score} dictionary.
    """

    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def compute(self, data: DataView) -> Dict[str, float]:
        """
        Compute factor scores for all available symbols.

        Omit symbols with insufficient data rather than returning NaN or zero.

        Args:
            data: DataView at current timestamp

        Returns:
            {symbol: score} dictionary
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
