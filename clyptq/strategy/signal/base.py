"""Base Signal class - unified foundation for Alpha and Factor.

Signal is the fundamental unit for all operations that return an N×T matrix.
Alpha and Factor have the same structure, differing only in their role.

Core principles:
- Same structure: All Signals return DataFrame (T×N)
- Same operations: All operators applicable (rank, zscore, neutralize...)
- Only role differs: SignalRole distinguishes Alpha/Factor/Intermediate

Classes:
- SignalRole: Role classification of Signal (ALPHA, FACTOR, INTERMEDIATE)
- BaseSignal: Abstract base class for all Signals
- Signal: Wrapper class with pipe() support

Example:
    ```python
    from clyptq.strategy.signal.base import BaseSignal, SignalRole
    from clyptq import operator

    class MomentumSignal(BaseSignal):
        role = SignalRole.FACTOR  # or ALPHA

        def compute(self, data):
            close = data.close
            return operator.ts_returns(close, self.lookback)
    ```
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import pandas as pd

if TYPE_CHECKING:
    from clyptq.strategy.transform.base import BaseTransform
    from clyptq.universe.loader import UniverseData
    from clyptq.data.provider import DataProvider


class SignalRole(Enum):
    """Role classification of Signal.

    Role indicates how the Signal is used in the pipeline.
    - ALPHA: Return-generating signal (final betting target)
    - FACTOR: Risk-explaining characteristic (neutralization basis)
    - INTERMEDIATE: Intermediate calculation result (reusable block)
    """
    ALPHA = "alpha"
    FACTOR = "factor"
    INTERMEDIATE = "intermediate"


class BaseSignal(ABC):
    """Abstract base class for all Signals.

    Both Alpha and Factor inherit from this class.
    Structure is identical, with role attribute distinguishing the purpose.

    Attributes:
        role: Signal role (ALPHA, FACTOR, INTERMEDIATE) - used for config/storage
        name: Signal name
        params: Parameter dictionary
        lookback: Window size required for calculation

    Example:
        ```python
        class MyAlpha(BaseSignal):
            role = SignalRole.ALPHA
            default_params = {"window": 20}

            def compute(self, data):
                return operator.ts_mean(data.close, self.lookback)

        class MyFactor(BaseSignal):
            role = SignalRole.FACTOR

            def compute(self, data):
                return operator.zscore(operator.ts_std(data.returns, self.lookback))
        ```
    """

    # Class-level defaults (override in subclasses)
    role: SignalRole = SignalRole.INTERMEDIATE
    default_params: Dict[str, Any] = {}

    def __init__(
        self,
        name: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        lookback: int = 20,
        role: Optional[SignalRole] = None,
    ):
        """Initialize BaseSignal.

        Args:
            name: Signal name (default: class name)
            params: Parameter override
            lookback: Calculation window
            role: Signal role override (for config/storage)
        """
        self.name = name or self.__class__.__name__
        self.params = {**self.default_params, **(params or {})}
        self.lookback = lookback

        # Role setting (can be overridden at instance level)
        if role is not None:
            self.role = role

    @abstractmethod
    def compute(
        self,
        data: Union["UniverseData", "DataProvider", Dict[str, pd.DataFrame]],
    ) -> pd.DataFrame:
        """Calculate Signal values.

        This method is the core calculation logic.
        Must be implemented using operators.

        Args:
            data: Input data
                - UniverseData: Access via attributes like close, volume
                - DataProvider: Access via data['close'], etc.
                - Dict: {'close': df, 'volume': df, ...}

        Returns:
            Signal values in DataFrame (T×N) format
        """
        pass

    def warmup_periods(self) -> int:
        """Number of periods required for warmup.

        Returns:
            Warmup period (default: lookback)
        """
        if self.params:
            return max(self.lookback, max(self.params.values(), default=0))
        return self.lookback

    @property
    def is_alpha(self) -> bool:
        """Check if this is an Alpha role."""
        return self.role == SignalRole.ALPHA

    @property
    def is_factor(self) -> bool:
        """Check if this is a Factor role."""
        return self.role == SignalRole.FACTOR

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"role={self.role.value}, "
            f"lookback={self.lookback})"
        )


class Signal:
    """Signal wrapper class with pipe() support.

    Wraps BaseSignal or raw DataFrame and
    allows applying transform pipelines.

    Example:
        ```python
        # Wrapping BaseSignal
        signal = Signal(MomentumSignal(lookback=20))
            .pipe(Demean())
            .pipe(L1Norm())

        # Wrapping Raw DataFrame
        signal = Signal(my_dataframe).pipe(ZScoreNormalizer())

        # Accessing values
        result = signal.value
        ```
    """

    def __init__(
        self,
        source: Union[pd.DataFrame, pd.Series, BaseSignal],
        data_provider: Optional["DataProvider"] = None,
        name: Optional[str] = None,
    ):
        """Initialize Signal.

        Args:
            source: BaseSignal instance or raw DataFrame/Series
            data_provider: DataProvider required for Transform
            name: Signal name
        """
        self._transforms: List["BaseTransform"] = []
        self._data_provider = data_provider

        if isinstance(source, BaseSignal):
            self._base_signal = source
            self._data: Optional[pd.DataFrame] = None
            self.name = name or source.name
            self.role = source.role
        elif isinstance(source, (pd.DataFrame, pd.Series)):
            self._base_signal = None
            self._data = source if isinstance(source, pd.DataFrame) else source.to_frame().T
            self.name = name or "RawSignal"
            self.role = SignalRole.INTERMEDIATE
        else:
            raise TypeError(
                f"Signal expects DataFrame, Series, or BaseSignal. Got {type(source)}"
            )

    def pipe(self, *transforms: "BaseTransform") -> "Signal":
        """Apply Transform pipeline (fluent API).

        Args:
            *transforms: Transforms to apply

        Returns:
            New Signal (with transforms applied)

        Example:
            signal.pipe(Demean(), Winsorizer(0.05, 0.95), L1Norm())
        """
        new_signal = Signal.__new__(Signal)
        new_signal._base_signal = self._base_signal
        new_signal._data = self._data.copy() if self._data is not None else None
        new_signal._transforms = self._transforms.copy()
        new_signal._data_provider = self._data_provider
        new_signal.name = self.name
        new_signal.role = self.role

        for transform in transforms:
            new_signal._transforms.append(transform)
            if new_signal._data is not None:
                result = transform.compute(new_signal._data)
                if isinstance(result, pd.DataFrame):
                    new_signal._data = result
                elif isinstance(result, pd.Series):
                    new_signal._data = result.to_frame().T

        return new_signal

    @property
    def value(self) -> pd.DataFrame:
        """Signal values with Transform applied.

        Returns:
            DataFrame (T×N)

        Raises:
            RuntimeError: When compute() not called in BaseSignal mode
        """
        if self._data is not None:
            return self._data

        raise RuntimeError(
            "Signal with BaseSignal needs data to compute. "
            "Use signal.compute(data) first."
        )

    def compute(
        self,
        data: Union["UniverseData", "DataProvider", Dict[str, pd.DataFrame]],
    ) -> "Signal":
        """Calculate Signal values using BaseSignal.

        Args:
            data: Input data

        Returns:
            self (computed)
        """
        if self._base_signal is None:
            return self

        # Set data provider for transforms
        if hasattr(data, '__getitem__'):
            self._data_provider = data

        # Compute raw signal
        scores = self._base_signal.compute(data)

        if scores.empty:
            self._data = pd.DataFrame(dtype=float)
            return self

        # Apply transforms
        for transform in self._transforms:
            if hasattr(transform, 'compute'):
                result = transform.compute(scores)
                if isinstance(result, pd.DataFrame):
                    scores = result
                elif isinstance(result, pd.Series):
                    scores = result.to_frame().T

        self._data = scores
        return self

    def with_provider(self, data_provider: "DataProvider") -> "Signal":
        """Set DataProvider.

        Args:
            data_provider: DataProvider instance

        Returns:
            self (for chaining)
        """
        self._data_provider = data_provider
        return self

    def warmup_periods(self) -> int:
        """Return warmup period."""
        if self._base_signal is not None:
            return self._base_signal.warmup_periods()
        return 0

    def __repr__(self) -> str:
        status = "computed" if self._data is not None else "pending"
        return (
            f"Signal(name={self.name}, "
            f"role={self.role.value}, "
            f"transforms={len(self._transforms)}, "
            f"status={status})"
        )
