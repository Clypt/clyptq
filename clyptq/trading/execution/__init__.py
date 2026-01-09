"""Order execution layer.

Executors:
- Executor: Base ABC for all executors
- BacktestExecutor: Deterministic simulation executor
- LiveExecutor: ABC for live trading executors
- CCXTExecutor: Crypto exchanges via CCXT
- MultiAssetRouter: Routes orders to appropriate executors by symbol
"""

from clyptq.trading.execution.base import Executor
from clyptq.trading.execution.backtest import BacktestExecutor
from clyptq.trading.execution.live import LiveExecutor
from clyptq.trading.execution.router import MultiAssetRouter

# Optional ccxt
try:
    from clyptq.trading.execution.live import CCXTExecutor
except ImportError:
    CCXTExecutor = None
from clyptq.trading.execution.costs import (
    apply_slippage,
    calculate_fee,
    estimate_fill_cost,
    estimate_transaction_costs,
    create_fill_from_order,
    calculate_turnover,
    estimate_rebalance_cost,
)
from clyptq.trading.execution.portfolio import (
    compute_target_positions,
    compute_order_deltas,
    notional_to_quantity,
    quantity_to_notional,
    compute_turnover as compute_portfolio_turnover,
    apply_turnover_constraint,
    apply_position_limits,
)

__all__ = [
    # Executors
    "Executor",
    "BacktestExecutor",
    "LiveExecutor",
    "CCXTExecutor",
    "MultiAssetRouter",
    # Cost functions
    "apply_slippage",
    "calculate_fee",
    "estimate_fill_cost",
    "estimate_transaction_costs",
    "create_fill_from_order",
    "calculate_turnover",
    "estimate_rebalance_cost",
    # Portfolio position helpers
    "compute_target_positions",
    "compute_order_deltas",
    "notional_to_quantity",
    "quantity_to_notional",
    "compute_portfolio_turnover",
    "apply_turnover_constraint",
    "apply_position_limits",
]
