import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from clypt.types import Fill, OrderSide, Position, Snapshot


class PortfolioState:
    """Portfolio state tracking with cash constraint and overselling prevention."""

    def __init__(self, initial_cash: float = 10000.0):
        if initial_cash <= 0:
            raise ValueError(f"initial_cash must be positive, got {initial_cash}")

        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self._trades_count = 0
        self._cumulative_realized_pnl = 0.0

    def apply_fill(self, fill: Fill) -> None:
        trade_value = fill.amount * fill.price

        if fill.side == OrderSide.BUY:
            # CRITICAL CHECK 1: Cash constraint
            required_cash = trade_value + fill.fee
            if self.cash < required_cash:
                raise ValueError(
                    f"Insufficient cash for {fill.symbol}: "
                    f"need {required_cash:.2f}, have {self.cash:.2f}"
                )

            # Deduct cash
            self.cash -= required_cash

            # Update position
            if fill.symbol in self.positions:
                pos = self.positions[fill.symbol]
                # Update average price
                total_cost = pos.amount * pos.avg_price + trade_value
                total_amount = pos.amount + fill.amount
                pos.avg_price = total_cost / total_amount if total_amount > 0 else fill.price
                pos.amount = total_amount
            else:
                # New position
                self.positions[fill.symbol] = Position(
                    symbol=fill.symbol,
                    amount=fill.amount,
                    avg_price=fill.price,
                )

        else:  # SELL
            # CRITICAL CHECK 2: Overselling prevention
            current_amount = (
                self.positions[fill.symbol].amount
                if fill.symbol in self.positions
                else 0.0
            )

            if fill.amount > current_amount + 1e-8:  # Small tolerance for floating point
                raise ValueError(
                    f"Overselling {fill.symbol}: "
                    f"trying to sell {fill.amount:.4f}, only have {current_amount:.4f}"
                )

            # Add cash (minus fee)
            self.cash += trade_value - fill.fee

            # Update position
            if fill.symbol in self.positions:
                pos = self.positions[fill.symbol]

                # Calculate realized P&L
                realized_pnl = (fill.price - pos.avg_price) * fill.amount
                pos.realized_pnl += realized_pnl
                self._cumulative_realized_pnl += realized_pnl  # Track across all positions

                # Reduce position
                pos.amount -= fill.amount

                # Remove if fully closed
                if abs(pos.amount) < 1e-8:
                    del self.positions[fill.symbol]

        self._trades_count += 1

    def get_snapshot(
        self, timestamp: datetime, prices: Dict[str, float]
    ) -> Snapshot:
        """
        Get current portfolio snapshot.

        Args:
            timestamp: Current timestamp
            prices: Current market prices

        Returns:
            Portfolio snapshot with all metrics
        """
        # Update unrealized P&L for all positions
        positions_value = 0.0

        for symbol, pos in self.positions.items():
            if symbol in prices:
                current_price = prices[symbol]
                market_value = pos.amount * current_price
                positions_value += market_value

                # Update unrealized P&L
                pos.unrealized_pnl = (current_price - pos.avg_price) * pos.amount

        # Total equity
        equity = self.cash + positions_value

        # Create snapshot
        return Snapshot(
            timestamp=timestamp,
            equity=equity,
            cash=self.cash,
            positions=self.positions.copy(),
            positions_value=positions_value,
        )

    def get_weights(self, prices: Dict[str, float]) -> Dict[str, float]:
        """
        Get current portfolio weights.

        Args:
            prices: Current market prices

        Returns:
            Dictionary of {symbol: weight}
        """
        # Calculate equity
        positions_value = sum(
            pos.amount * prices.get(symbol, 0.0)
            for symbol, pos in self.positions.items()
        )
        equity = self.cash + positions_value

        if equity < 1e-8:
            return {}

        # Calculate weights
        weights = {}
        for symbol, pos in self.positions.items():
            if symbol in prices:
                market_value = pos.amount * prices[symbol]
                weights[symbol] = market_value / equity

        return weights

    def save_state(self, filepath: Path) -> None:
        """
        Save portfolio state to file for recovery.

        Args:
            filepath: Path to save state
        """
        state = {
            "initial_cash": self.initial_cash,
            "cash": self.cash,
            "positions": {
                symbol: {
                    "symbol": pos.symbol,
                    "amount": pos.amount,
                    "avg_price": pos.avg_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "realized_pnl": pos.realized_pnl,
                }
                for symbol, pos in self.positions.items()
            },
            "trades_count": self._trades_count,
            "cumulative_realized_pnl": self._cumulative_realized_pnl,
        }

        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load_state(cls, filepath: Path) -> "PortfolioState":
        """
        Load portfolio state from file.

        Args:
            filepath: Path to saved state

        Returns:
            Restored portfolio state
        """
        with open(filepath, "r") as f:
            state = json.load(f)

        portfolio = cls(initial_cash=state["initial_cash"])
        portfolio.cash = state["cash"]
        portfolio._trades_count = state.get("trades_count", 0)
        portfolio._cumulative_realized_pnl = state.get("cumulative_realized_pnl", 0.0)

        # Restore positions
        for symbol, pos_data in state["positions"].items():
            portfolio.positions[symbol] = Position(
                symbol=pos_data["symbol"],
                amount=pos_data["amount"],
                avg_price=pos_data["avg_price"],
                unrealized_pnl=pos_data.get("unrealized_pnl", 0.0),
                realized_pnl=pos_data.get("realized_pnl", 0.0),
            )

        return portfolio

    def reset(self) -> None:
        """Reset portfolio to initial state."""
        self.cash = self.initial_cash
        self.positions.clear()
        self._trades_count = 0
        self._cumulative_realized_pnl = 0.0

    @property
    def num_positions(self) -> int:
        """Get number of active positions."""
        return len(self.positions)

    @property
    def total_realized_pnl(self) -> float:
        """Get total realized P&L across all positions (including closed ones)."""
        return self._cumulative_realized_pnl

    @property
    def total_unrealized_pnl(self) -> float:
        """Get total unrealized P&L across all open positions."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PortfolioState(cash={self.cash:.2f}, "
            f"positions={self.num_positions}, "
            f"trades={self._trades_count})"
        )
