"""
Factor registry for managing and discovering factors.

Provides centralized registry for factor instances and factory methods.
"""

from typing import Dict, List, Optional, Type

from clypt.factors.base import Factor


class FactorRegistry:
    """
    Central registry for factor management.

    Allows registering, retrieving, and listing available factors.
    """

    def __init__(self):
        """Initialize empty registry."""
        self._factors: Dict[str, Factor] = {}
        self._factory: Dict[str, Type[Factor]] = {}

    def register(self, factor: Factor, name: Optional[str] = None) -> None:
        """
        Register a factor instance.

        Args:
            factor: Factor instance to register
            name: Registration name (defaults to factor.name)

        Raises:
            ValueError: If name already registered
        """
        reg_name = name or factor.name

        if reg_name in self._factors:
            raise ValueError(f"Factor '{reg_name}' already registered")

        self._factors[reg_name] = factor

    def register_class(self, factor_class: Type[Factor], name: Optional[str] = None) -> None:
        """
        Register a factor class for factory instantiation.

        Args:
            factor_class: Factor class to register
            name: Registration name (defaults to class name)

        Raises:
            ValueError: If name already registered
        """
        reg_name = name or factor_class.__name__

        if reg_name in self._factory:
            raise ValueError(f"Factor class '{reg_name}' already registered")

        self._factory[reg_name] = factor_class

    def get(self, name: str) -> Factor:
        """
        Get registered factor instance by name.

        Args:
            name: Factor name

        Returns:
            Factor instance

        Raises:
            KeyError: If factor not found
        """
        if name not in self._factors:
            raise KeyError(f"Factor '{name}' not found in registry")

        return self._factors[name]

    def create(self, name: str, **kwargs) -> Factor:
        """
        Create factor instance from registered class.

        Args:
            name: Factor class name
            **kwargs: Arguments to pass to factor constructor

        Returns:
            New factor instance

        Raises:
            KeyError: If factor class not found
        """
        if name not in self._factory:
            raise KeyError(f"Factor class '{name}' not found in registry")

        return self._factory[name](**kwargs)

    def list_factors(self) -> List[str]:
        """Get list of registered factor instance names."""
        return list(self._factors.keys())

    def list_classes(self) -> List[str]:
        """Get list of registered factor class names."""
        return list(self._factory.keys())

    def remove(self, name: str) -> None:
        """
        Remove factor from registry.

        Args:
            name: Factor name to remove

        Raises:
            KeyError: If factor not found
        """
        if name not in self._factors:
            raise KeyError(f"Factor '{name}' not found in registry")

        del self._factors[name]

    def clear(self) -> None:
        """Clear all registered factors and classes."""
        self._factors.clear()
        self._factory.clear()

    def __contains__(self, name: str) -> bool:
        """Check if factor is registered."""
        return name in self._factors or name in self._factory

    def __len__(self) -> int:
        """Get total number of registered factors and classes."""
        return len(self._factors) + len(self._factory)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"FactorRegistry("
            f"factors={len(self._factors)}, "
            f"classes={len(self._factory)})"
        )


# Global registry instance
_global_registry = FactorRegistry()


def register(factor: Factor, name: Optional[str] = None) -> None:
    """
    Register factor in global registry.

    Args:
        factor: Factor instance
        name: Registration name
    """
    _global_registry.register(factor, name)


def register_class(factor_class: Type[Factor], name: Optional[str] = None) -> None:
    """
    Register factor class in global registry.

    Args:
        factor_class: Factor class
        name: Registration name
    """
    _global_registry.register_class(factor_class, name)


def get_factor(name: str) -> Factor:
    """
    Get factor from global registry.

    Args:
        name: Factor name

    Returns:
        Factor instance
    """
    return _global_registry.get(name)


def create_factor(name: str, **kwargs) -> Factor:
    """
    Create factor instance from global registry.

    Args:
        name: Factor class name
        **kwargs: Constructor arguments

    Returns:
        New factor instance
    """
    return _global_registry.create(name, **kwargs)


def list_factors() -> List[str]:
    """Get list of all registered factors."""
    return _global_registry.list_factors()


def list_factor_classes() -> List[str]:
    """Get list of all registered factor classes."""
    return _global_registry.list_classes()


def get_global_registry() -> FactorRegistry:
    """Get global factor registry."""
    return _global_registry
