"""Portfolio state management.

PortfolioState: Spot cash/position state tracking
FuturesPortfolioState: Futures margin/leverage position tracking

- apply_fill(): Apply fill
- get_snapshot(): Create snapshot
- get_weights(): Current weights
- equity(): Calculate total equity

Note: Order calculation (target position, order delta) is in trading/execution/portfolio.py
"""

from clyptq.trading.portfolio.state import PortfolioState, FuturesPortfolioState

__all__ = ["PortfolioState", "FuturesPortfolioState"]
