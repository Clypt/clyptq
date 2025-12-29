import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from clyptq.types import Fill, OrderSide, Position, Snapshot


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
            required_cash = trade_value + fill.fee
            if self.cash < required_cash:
                raise ValueError(
                    f"Insufficient cash for {fill.symbol}: "
                    f"need {required_cash:.2f}, have {self.cash:.2f}"
                )

            self.cash -= required_cash

            if fill.symbol in self.positions:
                pos = self.positions[fill.symbol]
                total_cost = pos.amount * pos.avg_price + trade_value
                total_amount = pos.amount + fill.amount
                pos.avg_price = total_cost / total_amount if total_amount > 0 else fill.price
                pos.amount = total_amount
            else:
                self.positions[fill.symbol] = Position(
                    symbol=fill.symbol,
                    amount=fill.amount,
                    avg_price=fill.price,
                )

        else:
            current_amount = (
                self.positions[fill.symbol].amount
                if fill.symbol in self.positions
                else 0.0
            )

            if fill.amount > current_amount + 1e-8:
                raise ValueError(
                    f"Overselling {fill.symbol}: "
                    f"trying to sell {fill.amount:.4f}, only have {current_amount:.4f}"
                )

            self.cash += trade_value - fill.fee

            if fill.symbol in self.positions:
                pos = self.positions[fill.symbol]

                realized_pnl = (fill.price - pos.avg_price) * fill.amount
                pos.realized_pnl += realized_pnl
                self._cumulative_realized_pnl += realized_pnl

                pos.amount -= fill.amount

                if abs(pos.amount) < 1e-8:
                    del self.positions[fill.symbol]

        self._trades_count += 1

    def get_snapshot(
        self, timestamp: datetime, prices: Dict[str, float]
    ) -> Snapshot:
        positions_value = 0.0

        for symbol, pos in self.positions.items():
            if symbol in prices:
                current_price = prices[symbol]
                market_value = pos.amount * current_price
                positions_value += market_value
                pos.unrealized_pnl = (current_price - pos.avg_price) * pos.amount

        equity = self.cash + positions_value

        return Snapshot(
            timestamp=timestamp,
            equity=equity,
            cash=self.cash,
            positions=self.positions.copy(),
            positions_value=positions_value,
        )

    def get_weights(self, prices: Dict[str, float]) -> Dict[str, float]:
        positions_value = sum(
            pos.amount * prices.get(symbol, 0.0)
            for symbol, pos in self.positions.items()
        )
        equity = self.cash + positions_value

        if equity < 1e-8:
            return {}

        weights = {}
        for symbol, pos in self.positions.items():
            if symbol in prices:
                market_value = pos.amount * prices[symbol]
                weights[symbol] = market_value / equity

        return weights

    def save_state(self, filepath: Path) -> None:
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
        self.cash = self.initial_cash
        self.positions.clear()
        self._trades_count = 0
        self._cumulative_realized_pnl = 0.0

    @property
    def num_positions(self) -> int:
        return len(self.positions)

    @property
    def total_realized_pnl(self) -> float:
        return self._cumulative_realized_pnl

    @property
    def total_unrealized_pnl(self) -> float:
        return sum(pos.unrealized_pnl for pos in self.positions.values())

    def __repr__(self) -> str:
        return (
            f"PortfolioState(cash={self.cash:.2f}, "
            f"positions={self.num_positions}, "
            f"trades={self._trades_count})"
        )
