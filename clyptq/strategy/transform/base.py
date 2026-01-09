"""Base transform interface."""

from abc import ABC, abstractmethod


class BaseTransform(ABC):
    """Base class for all transforms."""

    @abstractmethod
    def compute(self, data):
        """Apply transform to data."""
        pass
