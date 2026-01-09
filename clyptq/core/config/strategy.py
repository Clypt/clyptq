"""Strategy configuration for serialization and reproducibility.

StrategyConfig captures:
- Multiple alpha configurations
- Combination method
- Portfolio-level transforms
- Universe settings (reference only)

Note:
    - Combiner class has been removed. Use combination_method instead.
    - SimpleStrategy removed. build() creates a dynamic Strategy subclass.
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from clyptq.strategy.base import Strategy

from clyptq.core.config.alpha import AlphaConfig, TransformConfig


@dataclass
class StrategyConfig:
    """Configuration for a complete strategy.

    Captures:
    - Multiple alpha configurations
    - Combination method and weights
    - Portfolio-level transforms
    - Rebalancing settings

    Example:
        ```python
        config = StrategyConfig(
            alphas=[momentum_config, reversal_config],
            combination_method="weighted_sum",
            combination_weights=[0.6, 0.4],
            transforms=[TransformConfig("Demean")],
            rebalance_freq="D",
            name="MultiFactorStrategy",
        )

        config.save("strategy_config.json")
        loaded = StrategyConfig.load("strategy_config.json")
        strategy = loaded.build(universe)
        ```
    """
    alphas: List[AlphaConfig] = field(default_factory=list)
    combination_method: str = "equal_weight"  # equal_weight, weighted_sum, rank_average, ic_weight
    combination_weights: Optional[List[float]] = None
    transforms: List[TransformConfig] = field(default_factory=list)
    rebalance_freq: str = "D"
    warmup: int = 50
    name: str = "Strategy"
    version: str = "1.0"

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "alphas": [a.to_dict() for a in self.alphas],
            "combination_method": self.combination_method,
            "combination_weights": self.combination_weights,
            "transforms": [t.to_dict() for t in self.transforms],
            "rebalance_freq": self.rebalance_freq,
            "warmup": self.warmup,
            "name": self.name,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StrategyConfig":
        """Deserialize from dict."""
        alphas = [AlphaConfig.from_dict(a) for a in d.get("alphas", [])]

        # Handle legacy combiner config (migration)
        combination_method = d.get("combination_method", "equal_weight")
        combination_weights = d.get("combination_weights")

        if d.get("combiner"):
            warnings.warn(
                "Legacy 'combiner' field detected. Migrating to combination_method. "
                "Please update your config files.",
                DeprecationWarning,
            )
            combiner = d["combiner"]
            combination_method = combiner.get("method", "equal_weight")
            combination_weights = list(combiner.get("weights", {}).values()) if combiner.get("weights") else None

        transforms = [
            TransformConfig.from_dict(t) for t in d.get("transforms", [])
        ]

        return cls(
            alphas=alphas,
            combination_method=combination_method,
            combination_weights=combination_weights,
            transforms=transforms,
            rebalance_freq=d.get("rebalance_freq", "D"),
            warmup=d.get("warmup", 50),
            name=d.get("name", "Strategy"),
            version=d.get("version", "1.0"),
        )

    def save(self, path: str) -> None:
        """Save config to JSON file."""
        import json
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "StrategyConfig":
        """Load config from JSON file."""
        import json
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))

    def build(self, universe: "Any") -> "Strategy":
        """Build Strategy instance from config.

        Creates a dynamic Strategy subclass with the configured alphas,
        transforms, and combination method.

        Args:
            universe: Universe instance (required, not serializable)

        Returns:
            Strategy instance
        """
        from clyptq.strategy.base import Strategy
        from clyptq.operator.linalg import ca_reduce_avg, ca_weighted_sum

        # Build alphas and transforms
        alphas = [a.build() for a in self.alphas]
        transforms = [t.build() for t in self.transforms]

        config = self

        class ConfiguredStrategy(Strategy):
            """Dynamically created strategy from config."""

            def __init__(self):
                self.__class__.universe = universe
                self.__class__.rebalance_freq = config.rebalance_freq
                super().__init__(name=config.name)
                self._alphas = alphas
                self._transforms = transforms
                self._combination_method = config.combination_method
                self._combination_weights = config.combination_weights

            def warmup_periods(self) -> int:
                return config.warmup

            def compute_signal(self) -> pd.DataFrame:
                if not self._alphas:
                    return pd.DataFrame()

                # Compute each alpha
                signals = []
                for alpha in self._alphas:
                    sig = alpha.compute(self.provider)
                    if sig is not None and not sig.empty:
                        signals.append(sig)

                if not signals:
                    return pd.DataFrame()

                # Combine based on method
                if self._combination_method == "weighted_sum" and self._combination_weights:
                    combined = ca_weighted_sum(*signals, weights=self._combination_weights)
                else:
                    combined = ca_reduce_avg(*signals)

                # Apply transforms
                for transform in self._transforms:
                    combined = transform.apply(combined, self.provider)

                return combined

        return ConfiguredStrategy()
