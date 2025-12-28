"""
Configuration management for the Clypt Trading Engine.

Loads configuration from environment variables and .env files,
with sensible defaults for all settings.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from clypt.types import Constraints, CostModel, EngineMode


# Load .env file if present
load_dotenv()


@dataclass
class ExchangeConfig:
    """Exchange API configuration."""

    exchange_id: str
    api_key: str
    api_secret: str
    sandbox: bool = True

    @classmethod
    def from_env(cls, exchange_id: str = "binance") -> "ExchangeConfig":
        """Load exchange config from environment variables."""
        return cls(
            exchange_id=exchange_id,
            api_key=os.getenv(f"{exchange_id.upper()}_API_KEY", ""),
            api_secret=os.getenv(f"{exchange_id.upper()}_API_SECRET", ""),
            sandbox=os.getenv(f"{exchange_id.upper()}_SANDBOX", "true").lower() == "true",
        )


@dataclass
class DataConfig:
    """Data storage configuration."""

    data_dir: Path
    cache_dir: Path
    use_cache: bool = True

    @classmethod
    def from_env(cls) -> "DataConfig":
        """Load data config from environment variables."""
        data_dir = Path(os.getenv("CLYPT_DATA_DIR", "./data"))
        cache_dir = Path(os.getenv("CLYPT_CACHE_DIR", "./cache"))
        use_cache = os.getenv("CLYPT_USE_CACHE", "true").lower() == "true"

        # Create directories if they don't exist
        data_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)

        return cls(data_dir=data_dir, cache_dir=cache_dir, use_cache=use_cache)


@dataclass
class EngineConfiguration:
    """Complete engine configuration."""

    mode: EngineMode
    initial_capital: float
    cost_model: CostModel
    constraints: Constraints
    rebalance_schedule: str
    exchange: Optional[ExchangeConfig] = None
    data: Optional[DataConfig] = None

    @classmethod
    def from_env(cls, mode: Optional[str] = None) -> "EngineConfiguration":
        """
        Load complete configuration from environment variables.

        Args:
            mode: Engine mode ("backtest", "paper", "live"). If None, reads from CLYPT_MODE env var.

        Returns:
            Complete engine configuration.
        """
        # Engine mode
        mode_str = mode or os.getenv("CLYPT_MODE", "backtest")
        engine_mode = EngineMode(mode_str)

        # Capital
        initial_capital = float(os.getenv("CLYPT_INITIAL_CAPITAL", "10000.0"))

        # Cost model
        cost_model = CostModel(
            maker_fee=float(os.getenv("CLYPT_MAKER_FEE", "0.001")),
            taker_fee=float(os.getenv("CLYPT_TAKER_FEE", "0.001")),
            slippage_bps=float(os.getenv("CLYPT_SLIPPAGE_BPS", "5.0")),
        )

        # Constraints
        constraints = Constraints(
            max_position_size=float(os.getenv("CLYPT_MAX_POSITION_SIZE", "0.2")),
            max_gross_exposure=float(os.getenv("CLYPT_MAX_GROSS_EXPOSURE", "1.0")),
            min_position_size=float(os.getenv("CLYPT_MIN_POSITION_SIZE", "0.01")),
            max_num_positions=int(os.getenv("CLYPT_MAX_NUM_POSITIONS", "20")),
            allow_short=os.getenv("CLYPT_ALLOW_SHORT", "false").lower() == "true",
        )

        # Rebalance schedule
        rebalance_schedule = os.getenv("CLYPT_REBALANCE_SCHEDULE", "daily")
        if rebalance_schedule not in ("daily", "weekly", "monthly"):
            raise ValueError(
                f"Invalid rebalance_schedule: {rebalance_schedule}. "
                f"Must be 'daily', 'weekly', or 'monthly'"
            )

        # Exchange config (only for paper/live modes)
        exchange = None
        if engine_mode in (EngineMode.PAPER, EngineMode.LIVE):
            exchange_id = os.getenv("CLYPT_EXCHANGE", "binance")
            exchange = ExchangeConfig.from_env(exchange_id)

        # Data config
        data = DataConfig.from_env()

        return cls(
            mode=engine_mode,
            initial_capital=initial_capital,
            cost_model=cost_model,
            constraints=constraints,
            rebalance_schedule=rebalance_schedule,
            exchange=exchange,
            data=data,
        )

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "mode": self.mode.value,
            "initial_capital": self.initial_capital,
            "cost_model": {
                "maker_fee": self.cost_model.maker_fee,
                "taker_fee": self.cost_model.taker_fee,
                "slippage_bps": self.cost_model.slippage_bps,
            },
            "constraints": {
                "max_position_size": self.constraints.max_position_size,
                "max_gross_exposure": self.constraints.max_gross_exposure,
                "min_position_size": self.constraints.min_position_size,
                "max_num_positions": self.constraints.max_num_positions,
                "allow_short": self.constraints.allow_short,
            },
            "rebalance_schedule": self.rebalance_schedule,
            "exchange": {
                "exchange_id": self.exchange.exchange_id,
                "sandbox": self.exchange.sandbox,
            }
            if self.exchange
            else None,
            "data": {
                "data_dir": str(self.data.data_dir),
                "cache_dir": str(self.data.cache_dir),
                "use_cache": self.data.use_cache,
            }
            if self.data
            else None,
        }


# Convenience function for quick config loading
def load_config(mode: Optional[str] = None) -> EngineConfiguration:
    """
    Load configuration from environment variables.

    Args:
        mode: Engine mode override ("backtest", "paper", "live")

    Returns:
        Complete engine configuration

    Example:
        >>> config = load_config("backtest")
        >>> config.initial_capital
        10000.0
    """
    return EngineConfiguration.from_env(mode)
