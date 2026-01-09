"""Alpha 101 implementations.

This module contains 101 alpha factors from the WorldQuant Alpha 101 paper.
These are classic examples of quantitative alpha signals that can be used
for research, benchmarking, and as building blocks for new strategies.

Each alpha is implemented as a class inheriting from BaseSignal with:
- compute() method that produces the signal
- role = SignalRole.ALPHA
- default_params for tunable parameters

Usage:
    from clyptq.strategy.signal.alpha.library.alpha_101 import alpha_101_001

    alpha = alpha_101_001()
    signal = alpha.compute(ohlcv_data)
"""

from clyptq.strategy.signal.base import BaseSignal

# Legacy alias
BaseAlpha = BaseSignal

# Dynamically import all alpha classes
import importlib
import pkgutil
import os

_alpha_classes = {}

# Get the directory of this package
_package_dir = os.path.dirname(__file__)

# Import all alpha_101_XXX modules (001-101)
for _, module_name, _ in pkgutil.iter_modules([_package_dir]):
    if module_name.startswith('alpha_101_'):
        try:
            module = importlib.import_module(f'.{module_name}', __package__)
            # Find the alpha class in the module
            for attr_name in dir(module):
                if attr_name.startswith('alpha_101_'):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type) and issubclass(attr, BaseSignal) and attr != BaseSignal:
                        _alpha_classes[attr_name] = attr
                        globals()[attr_name] = attr
        except ImportError as e:
            print(f"Warning: Could not import {module_name}: {e}")

__all__ = ['BaseSignal', 'BaseAlpha'] + list(_alpha_classes.keys())
