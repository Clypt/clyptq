"""Alpha configuration for serialization and reproducibility.

Contains:
- AlphaConfig: Alpha configuration
- TransformConfig: Transform configuration
- Registry functions for alpha/transform types

For CombinerConfig, see core/config/combiner.py
For StrategyConfig, see core/config/strategy.py
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from clyptq.strategy.signal import BaseSignal
    from clyptq.strategy.transform.base import BaseTransform


# Registries for alpha, transform, and combiner types
ALPHA_REGISTRY: Dict[str, type] = {}
TRANSFORM_REGISTRY: Dict[str, type] = {}


def register_alpha(name: str):
    """Decorator to register an alpha class."""
    def decorator(cls):
        ALPHA_REGISTRY[name] = cls
        return cls
    return decorator


def register_transform(name: str):
    """Decorator to register a transform class."""
    def decorator(cls):
        TRANSFORM_REGISTRY[name] = cls
        return cls
    return decorator


@dataclass
class TransformConfig:
    """Configuration for a single transform.

    Example:
        ```python
        config = TransformConfig(
            type="FactorNeutralizer",
            params={"factors": ["Momentum", "Volatility"]}
        )
        transform = config.build()
        ```
    """
    type: str
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {"type": self.type, "params": self.params}

    @classmethod
    def from_dict(cls, d: dict) -> "TransformConfig":
        """Deserialize from dict."""
        return cls(type=d["type"], params=d.get("params", {}))

    def build(self) -> "BaseTransform":
        """Build transform instance from config."""
        if self.type not in TRANSFORM_REGISTRY:
            raise ValueError(f"Unknown transform type: {self.type}")
        return TRANSFORM_REGISTRY[self.type](**self.params)


@dataclass
class AlphaConfig:
    """Configuration for an alpha signal with transforms.

    Captures the complete specification for reproducibility:
    - Alpha computation logic
    - Transform pipeline
    - Pasteurize settings

    Example:
        ```python
        config = AlphaConfig(
            alpha_type="MomentumAlpha",
            alpha_params={"lookback": 20},
            transforms=[
                TransformConfig("Demean"),
                TransformConfig("FactorNeutralizer", {"factors": ["Mom", "Vol"]}),
            ],
            scope="GLOBAL",
            output="TRUNCATED",
            name="Momentum_Neutral",
        )

        # Save
        config.save("momentum_config.json")

        # Load and build
        loaded = AlphaConfig.load("momentum_config.json")
        alpha = loaded.build()
        ```

    Attributes:
        alpha_type: Registered alpha class name
        alpha_params: Parameters for alpha initialization
        transforms: List of transform configurations (applied in order)
        scope: "GLOBAL" (compute on N) or "LOCAL" (compute on n)
        output: "FULL" (return N) or "TRUNCATED" (return n)
        normalize: Whether to apply rank normalization at end
        name: Human-readable name for this configuration
        version: Config version for compatibility
    """
    alpha_type: str
    alpha_params: Dict[str, Any] = field(default_factory=dict)
    transforms: List[TransformConfig] = field(default_factory=list)
    scope: str = "GLOBAL"
    output: str = "TRUNCATED"
    normalize: bool = True
    name: str = "Alpha"
    version: str = "1.0"

    def to_dict(self) -> dict:
        """Serialize to dict for JSON/YAML storage."""
        return {
            "alpha_type": self.alpha_type,
            "alpha_params": self.alpha_params,
            "transforms": [t.to_dict() for t in self.transforms],
            "scope": self.scope,
            "output": self.output,
            "normalize": self.normalize,
            "name": self.name,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AlphaConfig":
        """Deserialize from dict."""
        transforms = [
            TransformConfig.from_dict(t) for t in d.get("transforms", [])
        ]
        return cls(
            alpha_type=d["alpha_type"],
            alpha_params=d.get("alpha_params", {}),
            transforms=transforms,
            scope=d.get("scope", "GLOBAL"),
            output=d.get("output", "TRUNCATED"),
            normalize=d.get("normalize", True),
            name=d.get("name", "Alpha"),
            version=d.get("version", "1.0"),
        )

    def save(self, path: str) -> None:
        """Save config to JSON file."""
        import json
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "AlphaConfig":
        """Load config from JSON file."""
        import json
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))

    def build(self) -> "BaseAlpha":
        """Build Alpha instance from config.

        Returns:
            Alpha instance with transforms applied
        """
        from clyptq.strategy.signal import Signal

        if self.alpha_type not in ALPHA_REGISTRY:
            raise ValueError(f"Unknown alpha type: {self.alpha_type}")

        # Build base alpha
        alpha_cls = ALPHA_REGISTRY[self.alpha_type]
        base_alpha = alpha_cls(**self.alpha_params)

        # Build transforms
        transforms = [t.build() for t in self.transforms]

        # Create Signal with transforms
        return Signal(
            base_alpha=base_alpha,
            transforms=transforms,
            scope=self.scope,
            output=self.output,
            normalize=self.normalize,
            name=self.name,
        )

    def copy(self, **overrides) -> "AlphaConfig":
        """Create a copy with optional overrides.

        Example:
            # Same alpha, different neutralization
            config2 = config.copy(
                transforms=[TransformConfig("Demean")],
                name="Momentum_DemeanOnly",
            )
        """
        d = self.to_dict()
        d.update(overrides)
        if "transforms" in overrides and isinstance(overrides["transforms"], list):
            d["transforms"] = [
                t.to_dict() if isinstance(t, TransformConfig) else t
                for t in overrides["transforms"]
            ]
        return AlphaConfig.from_dict(d)
