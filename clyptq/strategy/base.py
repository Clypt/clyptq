"""
Strategy class for complete trading strategies.

Architecture:
- Strategy declares: p (DataProvider), alphas, combiner, transforms
- Provider (p) is declared as class attribute with universe/data specs
- Engine binds source at runtime based on mode (backtest vs live)

Example:
    ```python
    from clyptq import Strategy, Engine
    from clyptq.data.provider import DataProvider
    from clyptq.data.spec import OHLCVSpec
    from clyptq.universe import DynamicUniverse, TopNFilter

    class MyStrategy(Strategy):
        # Provider declared as p (same variable name for research/strategy)
        p = DataProvider(
            universe=DynamicUniverse(
                base_symbols=["BTC", "ETH", "SOL", "AVAX", "MATIC"],
                filters=[TopNFilter(n=5, rank_by="volume")],
            ),
            data={"ohlcv": OHLCVSpec(timeframe="1h")},
            rebalance_freq="1d",
        )

        def warmup_periods(self) -> int:
            return 50

        def compute_signal(self):
            # Same code as research notebook
            close = p["close"]
            close_4h = p["close", "4h"]
            alpha = Alpha(rank(close.pct_change(12))).pipe(Demean(), L1Norm())
            return alpha.value

    # Backtest
    engine = Engine()
    result = engine.run(MyStrategy(), mode="backtest", start=..., end=...)
    ```
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from clyptq.operator import l1_norm, demean

if TYPE_CHECKING:
    from clyptq.data.provider import DataProvider
    from clyptq.data.spec import DataSpecType
    from clyptq.strategy.signal import Signal, BaseSignal
    from clyptq.strategy.transform.base import BaseTransform
    from clyptq.trading.portfolio.state import PortfolioState
    from clyptq.universe.base import BaseUniverse


class Strategy(ABC):
    """Base class for trading strategies.

    Subclasses declare their requirements as class attributes:
    - p: DataProvider (preferred) - same variable for research/strategy
    - universe: Universe defining tradeable symbols
    - data: Dict of {name: DataSpec} defining data requirements
    - rebalance_freq: Rebalancing frequency
    - alphas: List of Alpha instances
    - combiner: Combiner for combining alphas
    - transforms: List of transforms for portfolio construction

    The provider is injected by the runner at runtime.

    Attributes:
        p: DataProvider instance (class attribute or injected)
        provider: Alias for p
        name: Strategy name
    """

    # --- Class-level declarations (override in subclass) ---

    # DataProvider declaration (preferred way)
    # Usage: p = DataProvider(universe=..., data=..., rebalance_freq=...)
    p: Optional["DataProvider"] = None

    # Universe definition
    universe: Optional["BaseUniverse"] = None

    # Data requirements
    data: Dict[str, "DataSpecType"] = None

    # Rebalancing frequency
    rebalance_freq: str = "1d"

    # Strategy name
    name: str = None

    # Alpha, transform pipeline (optional - for declarative strategies)
    # Note: Combiner removed. Use operator.ca_* functions in compute_signal() instead.
    alphas: List["BaseSignal"] = None
    transforms: List["BaseTransform"] = None

    def __init__(self, name: Optional[str] = None):
        """Initialize Strategy.

        Auto-detects p (DataProvider) from class attribute.

        Args:
            name: Strategy name (defaults to class name)
        """
        self._name = name or self.__class__.name or self.__class__.__name__

        # Runtime state
        self._current_weights: Dict[str, float] = {}

        # Auto-detect provider from any class attribute (p, provider, etc.)
        self._provider: Optional["DataProvider"] = None
        self._provider_attr_name: Optional[str] = None  # Stores the attribute name of detected provider
        self._auto_detect_provider()

        # Initialize default data spec if not declared
        if self.__class__.data is None and self._provider is None:
            from clyptq.data.spec import OHLCVSpec
            self.__class__.data = {"ohlcv": OHLCVSpec()}

    def _auto_detect_provider(self) -> None:
        """Auto-detect DataProvider from any class attribute.

        Scans all class attributes to find a DataProvider instance.
        The attribute name can be anything (p, provider, data_provider, etc.).
        """
        from clyptq.data.provider import DataProvider

        # Scan all class attributes for DataProvider instance
        for attr_name in dir(self.__class__):
            # Skip private/dunder attributes
            if attr_name.startswith('_'):
                continue

            try:
                attr = getattr(self.__class__, attr_name, None)
            except AttributeError:
                continue

            if isinstance(attr, DataProvider):
                self._provider = attr
                self._provider_attr_name = attr_name  # Store attribute name for reference

                # Extract universe/data/rebalance_freq for Engine compatibility
                if hasattr(attr, 'universe') and attr.universe is not None:
                    self.__class__.universe = attr.universe
                if hasattr(attr, 'sources') and attr.sources:
                    self.__class__.data = attr.sources
                if hasattr(attr, 'rebalance_freq') and attr.rebalance_freq:
                    self.__class__.rebalance_freq = attr.rebalance_freq

                # Found provider, stop scanning
                break

    # --- Provider injection (called by runner) ---

    def bind_provider(self, provider: "DataProvider") -> "Strategy":
        """Bind a DataProvider to this strategy.

        Called by run_backtest/run_live to inject the provider.

        Args:
            provider: DataProvider instance (loaded with data)

        Returns:
            self for method chaining
        """
        self._provider = provider
        return self

    @property
    def provider(self) -> "DataProvider":
        """Access the bound DataProvider.

        Raises:
            RuntimeError: If provider not bound yet
        """
        if self._provider is None:
            raise RuntimeError(
                "Provider not bound. Use run_backtest() or run_live() to run the strategy, "
                "or call strategy.bind_provider(provider) manually."
            )
        return self._provider

    @property
    def is_bound(self) -> bool:
        """Check if provider is bound."""
        return self._provider is not None

    # --- Properties delegated to provider ---

    @property
    def symbols(self) -> List[str]:
        """Current symbols in universe."""
        return self.provider.universe_symbols

    @property
    def current_timestamp(self) -> Optional[datetime]:
        """Current timestamp from provider."""
        return self.provider.current_timestamp

    @property
    def in_universe(self) -> pd.Series:
        """Current universe membership."""
        return self.provider.in_universe

    # --- Signal computation (override in subclass) ---

    def compute_signal(self) -> pd.DataFrame:
        """Compute trading signal.

        Override this method for custom signal logic.
        Or define alphas/combiner for declarative approach.

        Returns:
            Signal as pd.DataFrame (T x N)
        """
        # If alphas defined, use declarative pipeline
        if self.alphas is not None:
            return self._compute_from_alphas()

        raise NotImplementedError(
            "Override compute_signal() or define alphas/combiner in subclass"
        )

    def _compute_from_alphas(self) -> pd.DataFrame:
        """Compute signal from declared alphas."""
        if not self.alphas:
            return pd.DataFrame()

        # Compute each alpha
        signals = []
        for alpha in self.alphas:
            sig = alpha.compute(self.provider)
            if sig is not None and not sig.empty:
                signals.append(sig)

        if not signals:
            return pd.DataFrame()

        # Combine signals using operator.ca_reduce_avg (simple equal weight)
        # For custom combination, override compute_signal() and use operator.ca_* functions
        from clyptq.operator.linalg import ca_reduce_avg
        combined = ca_reduce_avg(*signals)

        return combined

    def compute_weights(self) -> pd.Series:
        """Compute portfolio weights from signal.

        The signal returned from compute_signal() is used directly as weights.
        User is responsible for normalization (l1_norm, etc.) in compute_signal().

        Weight constraints:
        - Spot: abs(weights).sum() <= 1.0 (100% max exposure)
        - Futures: abs(weights).sum() <= leverage (e.g., 5x = 500% max)

        Returns:
            Portfolio weights as pd.Series (symbol -> weight)

        Raises:
            ValueError: If weights are not properly normalized
        """
        signal = self.compute_signal()

        if signal is None or (isinstance(signal, pd.DataFrame) and signal.empty):
            return pd.Series(dtype=float)

        # Get last row as Series
        if isinstance(signal, pd.DataFrame):
            weights = signal.iloc[-1]
        else:
            weights = signal

        # Apply transforms if defined
        if self.transforms:
            for transform in self.transforms:
                weights = transform.apply(weights, self.provider)

        # Validate weights
        abs_sum = weights.abs().sum()
        if abs_sum > 1.0 + 1e-6:  # Small tolerance for floating point
            raise ValueError(
                f"Weights abs sum ({abs_sum:.4f}) exceeds 1.0. "
                f"Use l1_norm() to normalize weights in compute_signal(). "
                f"For leverage, scale weights after normalization (e.g., l1_norm(signal) * leverage)."
            )

        return weights

    # --- Event handlers (called by Engine/Runner) ---

    def on_bar(self, timestamp: datetime) -> Dict[str, float]:
        """Called on each bar by Engine.

        Override for custom bar-level logic.

        Args:
            timestamp: Current timestamp

        Returns:
            Target weights {symbol: weight}
        """
        weights = self.compute_weights()
        self._current_weights = weights.to_dict() if not weights.empty else {}
        return self._current_weights

    def on_rebalance(self, timestamp: datetime) -> Dict[str, float]:
        """Called on rebalancing by Runner.

        Override for custom rebalancing logic.

        Args:
            timestamp: Current timestamp

        Returns:
            Target weights {symbol: weight}
        """
        return self.on_bar(timestamp)

    # --- Abstract methods ---

    @abstractmethod
    def warmup_periods(self) -> int:
        """Get required warmup periods (MUST override).

        This defines the lookback window needed for alpha computation.
        Used by:
        - Backtest: Skip first N bars for warmup
        - Live: Buffer size limit

        Returns:
            Number of bars needed for warmup

        Example:
            def warmup_periods(self) -> int:
                return 50  # 50-period MA
        """
        pass

    # --- Validation ---

    def validate(self) -> List[str]:
        """Validate strategy configuration.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        if not self.is_bound:
            errors.append("provider is not bound")
        elif not self._provider._loaded:
            errors.append("provider is not loaded")

        if self.universe is None and self.data is None:
            errors.append("either universe or data must be defined")

        return errors

    def __repr__(self) -> str:
        bound = "bound" if self.is_bound else "unbound"
        return f"{self.__class__.__name__}(name={self._name}, {bound})"
