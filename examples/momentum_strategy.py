"""
Example momentum trading strategy.

Demonstrates how to build a complete trading strategy using the Clypt Trading Engine.
"""

from datetime import datetime, timedelta
from typing import List

from clypt import Constraints, CostModel, EngineMode
from clypt.analytics.metrics import print_metrics
from clypt.data.loaders.ccxt_loader import load_crypto_data
from clypt.engine import BacktestExecutor, Engine
from clypt.factors.base import Factor, combine_factors
from clypt.factors.library.momentum import MomentumFactor, RSIFactor
from clypt.portfolio.construction import PortfolioConstructor, TopNConstructor
from clypt.strategy.base import Strategy


class MomentumStrategy(Strategy):
    """
    Simple momentum strategy.

    Combines price momentum and RSI factors to identify trending assets.
    Uses Top-N portfolio construction for equal-weight allocation.
    """

    def __init__(
        self,
        momentum_lookback: int = 20,
        rsi_lookback: int = 14,
        top_n: int = 5,
        max_position_size: float = 0.3,
        name: str = "Momentum",
    ):
        """
        Initialize momentum strategy.

        Args:
            momentum_lookback: Lookback period for momentum calculation
            rsi_lookback: Lookback period for RSI
            top_n: Number of top assets to hold
            max_position_size: Maximum position size per asset
            name: Strategy name
        """
        super().__init__(name)
        self.momentum_lookback = momentum_lookback
        self.rsi_lookback = rsi_lookback
        self.top_n = top_n
        self.max_position_size = max_position_size

    def factors(self) -> List[Factor]:
        """Return list of factors used by this strategy."""
        # Create momentum and RSI factors
        momentum = MomentumFactor(lookback=self.momentum_lookback, name="Momentum20")
        rsi = RSIFactor(lookback=self.rsi_lookback, name="RSI14")

        # Combine with equal weights
        combined = combine_factors([momentum, rsi], weights=[0.7, 0.3], name="Combined")

        return [combined]

    def portfolio_constructor(self) -> PortfolioConstructor:
        """Return portfolio constructor."""
        return TopNConstructor(top_n=self.top_n)

    def constraints(self) -> Constraints:
        """Return portfolio constraints."""
        return Constraints(
            max_position_size=self.max_position_size,
            max_gross_exposure=1.0,
            min_position_size=0.05,
            max_num_positions=self.top_n,
            allow_short=False,
        )

    def schedule(self) -> str:
        """Return rebalancing schedule."""
        return "daily"

    def warmup_periods(self) -> int:
        """Return warmup periods."""
        return max(self.momentum_lookback, self.rsi_lookback) + 10


def main():
    """Run example momentum strategy backtest."""
    print("=" * 70)
    print("CLYPT TRADING ENGINE - Momentum Strategy Example")
    print("=" * 70)

    # 1. Load data
    print("\n📊 Loading market data...")
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT"]

    store = load_crypto_data(
        symbols=symbols,
        exchange="binance",
        timeframe="1d",
        days=180,  # 6 months
    )

    print(f"Loaded {len(store)} symbols")

    # 2. Create strategy
    print("\n🎯 Creating momentum strategy...")
    strategy = MomentumStrategy(
        momentum_lookback=20,
        rsi_lookback=14,
        top_n=3,
        max_position_size=0.4,
    )

    print(f"Strategy: {strategy.name}")
    print(f"Factors: {[f.name for f in strategy.factors()]}")
    print(f"Constructor: {strategy.portfolio_constructor()}")

    # 3. Create engine
    print("\n⚙️  Initializing engine...")
    cost_model = CostModel(
        maker_fee=0.001,  # 0.1%
        taker_fee=0.001,
        slippage_bps=5.0,
    )

    executor = BacktestExecutor(cost_model)

    engine = Engine(
        strategy=strategy,
        data_store=store,
        mode=EngineMode.BACKTEST,
        executor=executor,
        initial_capital=10000.0,
    )

    # 4. Run backtest
    print("\n🚀 Running backtest...")
    date_range = store.get_date_range()

    # Use last 90 days for backtest
    end_date = date_range.end
    start_date = end_date - timedelta(days=90)

    result = engine.run_backtest(
        start=start_date, end=end_date, verbose=True
    )

    # 5. Display results
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)

    print_metrics(result.metrics)

    print(f"📈 Total trades: {len(result.trades)}")
    print(f"📊 Total snapshots: {len(result.snapshots)}")

    # Show final portfolio
    if result.snapshots:
        final_snapshot = result.snapshots[-1]
        print(f"\n💼 Final Portfolio:")
        print(f"  Cash: ${final_snapshot.cash:.2f}")
        print(f"  Positions Value: ${final_snapshot.positions_value:.2f}")
        print(f"  Total Equity: ${final_snapshot.equity:.2f}")
        print(f"  Number of Positions: {final_snapshot.num_positions}")

        if final_snapshot.positions:
            print(f"\n  Positions:")
            for symbol, pos in final_snapshot.positions.items():
                print(f"    {symbol}: {pos.amount:.4f} @ ${pos.avg_price:.2f}")

    print("\n" + "=" * 70)
    print("Backtest complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
