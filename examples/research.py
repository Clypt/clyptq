"""
Quantitative Research Example - Advanced Factor Testing

Tests multiple factor combinations:
1. Momentum-only (baseline)
2. Mean Reversion (contrarian)
3. Multi-Factor Blend
4. Long-Short Market Neutral

Usage:
    python examples/research.py
"""

from datetime import timedelta

from clyptq import CostModel, Constraints
from clyptq.analytics.risk.monte_carlo import MonteCarloSimulator
from clyptq.analytics.reporting.report import HTMLReportGenerator
from clyptq.analytics.performance.attribution import PerformanceAttributor
from clyptq.data.loaders.ccxt import load_crypto_data
from clyptq.trading.engine import BacktestEngine
from clyptq.trading.execution import BacktestExecutor
from clyptq.trading.factors.library.momentum import MomentumFactor
from clyptq.trading.factors.library.volatility import VolatilityFactor
from clyptq.trading.factors.library.volume import VolumeFactor
from clyptq.trading.factors.library.mean_reversion import BollingerFactor, ZScoreFactor
from clyptq.trading.factors.library.liquidity import AmihudFactor
from clyptq.trading.factors.ops.cross_sectional import normalize
from clyptq.trading.portfolio.constructors import TopNConstructor, ScoreWeightedConstructor
from clyptq.trading.strategy.base import SimpleStrategy


class MomentumOnlyStrategy(SimpleStrategy):
    """Baseline: Pure momentum (fails in bear market)."""

    def __init__(self):
        factors = [
            MomentumFactor(lookback=30),
            VolatilityFactor(lookback=30),
        ]
        constraints = Constraints(
            max_position_size=0.20,
            max_gross_exposure=1.0,
            min_position_size=0.08,
            max_num_positions=8,
            allow_short=False,
        )
        super().__init__(
            factors_list=factors,
            constructor=TopNConstructor(top_n=5),
            constraints_obj=constraints,
            schedule_str="weekly",
            warmup=35,
            name="MomentumOnly",
        )


class MeanReversionStrategy(SimpleStrategy):
    """
    Mean Reversion Strategy:
    - Buy oversold assets (Bollinger + ZScore)
    - High liquidity filter
    - Weekly rebalance
    - Works well in ranging/bear markets
    """

    def __init__(self):
        factors = [
            BollingerFactor(lookback=20, num_std=2.0),  # Buy when price < lower band
            ZScoreFactor(lookback=20),  # Buy when z-score < -1
            AmihudFactor(lookback=20),  # High liquidity
        ]
        constraints = Constraints(
            max_position_size=0.25,
            max_gross_exposure=0.9,  # Keep 10% cash
            min_position_size=0.10,
            max_num_positions=5,
            allow_short=False,
        )
        super().__init__(
            factors_list=factors,
            constructor=TopNConstructor(top_n=5),
            constraints_obj=constraints,
            schedule_str="weekly",
            warmup=25,
            name="MeanReversion",
        )

    def factors(self):
        """Apply normalization."""
        base_factors = super().factors()

        class NormalizedFactor:
            def __init__(self, base_factor):
                self.base_factor = base_factor
                self.lookback = getattr(base_factor, 'lookback', 20)

            def compute(self, view):
                scores = self.base_factor.compute(view)
                return normalize(scores)

        return [NormalizedFactor(f) for f in base_factors]


class MultiFactorStrategy(SimpleStrategy):
    """
    Multi-Factor Strategy:
    - Momentum (trend following)
    - Mean Reversion (contrarian)
    - Volatility (risk)
    - Volume (liquidity)
    - Liquidity (tradability)

    Combines 5 different signals for robust performance.
    """

    def __init__(self):
        factors = [
            MomentumFactor(lookback=30),  # Trend
            BollingerFactor(lookback=20, num_std=2.0),  # Mean reversion
            VolatilityFactor(lookback=30),  # Risk
            VolumeFactor(lookback=30),  # Activity
            AmihudFactor(lookback=20),  # Liquidity
        ]
        constraints = Constraints(
            max_position_size=0.20,
            max_gross_exposure=0.95,  # Keep 5% cash
            min_position_size=0.08,
            max_num_positions=7,
            allow_short=False,
        )
        super().__init__(
            factors_list=factors,
            constructor=ScoreWeightedConstructor(use_long_short=False),
            constraints_obj=constraints,
            schedule_str="weekly",
            warmup=35,
            name="MultiFactor",
        )

    def factors(self):
        """Apply normalization to all factors."""
        base_factors = super().factors()

        class NormalizedFactor:
            def __init__(self, base_factor):
                self.base_factor = base_factor
                self.lookback = getattr(base_factor, 'lookback', 20)

            def compute(self, view):
                scores = self.base_factor.compute(view)
                return normalize(scores)

        return [NormalizedFactor(f) for f in base_factors]


class LongShortMultiFactorStrategy(SimpleStrategy):
    """
    Long-Short Multi-Factor:
    - Long: Strong momentum + oversold + high liquidity
    - Short: Weak momentum + overbought + low liquidity
    - Market neutral (hedged)
    - Weekly rebalance
    """

    def __init__(self):
        factors = [
            MomentumFactor(lookback=30),
            BollingerFactor(lookback=20, num_std=2.0),
            VolatilityFactor(lookback=30),
            VolumeFactor(lookback=30),
            AmihudFactor(lookback=20),
        ]
        constraints = Constraints(
            max_position_size=0.20,
            max_gross_exposure=1.6,  # 80% long + 80% short
            min_position_size=0.05,
            max_num_positions=10,  # 5 longs + 5 shorts
            allow_short=True,
        )
        super().__init__(
            factors_list=factors,
            constructor=ScoreWeightedConstructor(use_long_short=True),
            constraints_obj=constraints,
            schedule_str="weekly",
            warmup=35,
            name="LongShortMulti",
        )

    def factors(self):
        """Apply normalization for long-short."""
        base_factors = super().factors()

        class NormalizedFactor:
            def __init__(self, base_factor):
                self.base_factor = base_factor
                self.lookback = getattr(base_factor, 'lookback', 20)

            def compute(self, view):
                scores = self.base_factor.compute(view)
                return normalize(scores)

        return [NormalizedFactor(f) for f in base_factors]


def run_backtest(strategy, store, start, end):
    """Run single backtest."""
    cost_model = CostModel(maker_fee=0.001, taker_fee=0.001, slippage_bps=5.0)
    executor = BacktestExecutor(cost_model)

    engine = BacktestEngine(
        strategy=strategy,
        data_store=store,
        executor=executor,
        initial_capital=100000.0,
    )

    return engine.run(start=start, end=end, verbose=False)


def print_comparison(results):
    """Print strategy comparison table."""
    print("\n" + "=" * 100)
    print(f"{'Strategy':<25} {'Return':<12} {'Sharpe':<10} {'MaxDD':<10} {'Trades':<8} {'Win%':<8}")
    print("=" * 100)

    for name, result in results.items():
        m = result.metrics
        print(
            f"{name:<25} "
            f"{m.total_return:>10.2%}  "
            f"{m.sharpe_ratio:>8.3f}  "
            f"{m.max_drawdown:>8.2%}  "
            f"{len(result.trades):>6}  "
            f"{m.win_rate:>6.2%}"
        )

    print("=" * 100)


def main():
    print("=" * 70)
    print("ADVANCED FACTOR RESEARCH - MEAN REVERSION vs MOMENTUM")
    print("=" * 70)

    # 1. Data (Top 20 cryptocurrencies by market cap)
    symbols = [
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
        "ADA/USDT", "AVAX/USDT", "DOGE/USDT", "DOT/USDT", "MATIC/USDT",
        "LTC/USDT", "LINK/USDT", "UNI/USDT", "ATOM/USDT", "XLM/USDT",
        "ALGO/USDT", "FIL/USDT", "NEAR/USDT", "APT/USDT", "ARB/USDT"
    ]
    print(f"\nLoading {len(symbols)} symbols (720 days)...")
    store = load_crypto_data(symbols=symbols, exchange="binance", timeframe="1d", days=720)

    date_range = store.get_date_range()
    start = date_range.end - timedelta(days=365)
    end = date_range.end
    print(f"Research period: {start.date()} to {end.date()}")

    # 2. Strategy Variants
    print("\n" + "=" * 70)
    print("TESTING 4 STRATEGY VARIANTS")
    print("=" * 70)

    strategies = {
        "1. MomentumOnly": MomentumOnlyStrategy(),
        "2. MeanReversion": MeanReversionStrategy(),
        "3. MultiFactor": MultiFactorStrategy(),
        "4. LongShortMulti": LongShortMultiFactorStrategy(),
    }

    results = {}
    for name, strategy in strategies.items():
        print(f"\nRunning {name}...")
        results[name] = run_backtest(strategy, store, start, end)

    print_comparison(results)

    # 3. Best Strategy Analysis
    print("\n" + "=" * 70)
    print("DETAILED ANALYSIS - BEST STRATEGY")
    print("=" * 70)

    # Find best by Sharpe
    best_name = max(results.items(), key=lambda x: x[1].metrics.sharpe_ratio)[0]
    best_result = results[best_name]
    m = best_result.metrics

    print(f"\nBest Strategy: {best_name}")
    print(f"\nPerformance Metrics:")
    print(f"  Total Return: {m.total_return:.2%}")
    print(f"  Annualized Return: {m.annualized_return:.2%}")
    print(f"  Sharpe Ratio: {m.sharpe_ratio:.3f}")
    print(f"  Sortino Ratio: {m.sortino_ratio:.3f}")
    calmar_ratio = m.annualized_return / m.max_drawdown if m.max_drawdown > 1e-10 else 0.0
    print(f"  Calmar Ratio: {calmar_ratio:.3f}")
    print(f"  Max Drawdown: {m.max_drawdown:.2%}")
    print(f"  Volatility: {m.volatility:.2%}")

    print(f"\nTrading Metrics:")
    print(f"  Total Trades: {len(best_result.trades)}")
    print(f"  Win Rate: {m.win_rate:.2%}")
    print(f"  Profit Factor: {m.profit_factor:.2f}")
    print(f"  Avg Trade P&L: ${m.avg_trade_pnl:.2f}")

    # 4. Performance Attribution
    print("\n" + "=" * 70)
    print("PERFORMANCE ATTRIBUTION")
    print("=" * 70)

    attributor = PerformanceAttributor()
    attribution = attributor.analyze(best_result)

    print(f"\nFactor Contribution:")
    if attribution.factor_attributions:
        for factor_attr in attribution.factor_attributions:
            print(f"  {factor_attr.factor_name}: {factor_attr.total_contribution:.2%}")
    else:
        print(f"  (Factor attribution not yet implemented)")

    print(f"\nTop 10 Asset Contributors:")
    sorted_assets = sorted(
        attribution.asset_attributions,
        key=lambda x: abs(x.weight_contribution),
        reverse=True
    )[:10]
    for asset_attr in sorted_assets:
        print(f"  {asset_attr.symbol}: {asset_attr.weight_contribution:.2%}")

    print(f"\nCost Breakdown:")
    print(f"  Transaction Costs: {attribution.transaction_cost_drag:.2%}")
    print(f"  Cash Drag: {attribution.cash_drag:.2%}")

    # 5. Monte Carlo Simulation
    print("\n" + "=" * 70)
    print("MONTE CARLO SIMULATION (1000 runs)")
    print("=" * 70)

    mc = MonteCarloSimulator(num_simulations=1000, random_seed=42)
    mc_result = mc.run(best_result)

    print(f"\nReturn Distribution:")
    print(f"  Mean: {mc_result.mean_return:.2%}")
    print(f"  Median: {mc_result.median_return:.2%}")
    print(f"  Std Dev: {mc_result.std_return:.2%}")
    print(f"  5th percentile: {mc_result.ci_5_return:.2%}")
    print(f"  95th percentile: {mc_result.ci_95_return:.2%}")

    print(f"\nRisk Metrics:")
    print(f"  Probability of Loss: {mc_result.probability_of_loss:.2%}")
    print(f"  Expected Shortfall (5%): {mc_result.expected_shortfall_5:.2%}")
    print(f"  Max DD (5th percentile): {mc_result.max_drawdown_5:.2%}")
    print(f"  Max DD (median): {mc_result.max_drawdown_50:.2%}")

    print(f"\nSharpe Confidence Interval:")
    print(f"  Mean: {mc_result.mean_sharpe:.3f}")
    print(f"  5th-95th: [{mc_result.ci_5_sharpe:.3f}, {mc_result.ci_95_sharpe:.3f}]")

    # 6. Strategy Comparison Summary
    print("\n" + "=" * 70)
    print("STRATEGY FACTOR COMPARISON")
    print("=" * 70)

    print(f"\n  1. MomentumOnly:")
    print(f"     Factors: Momentum(30d), Volatility(30d)")
    print(f"     Logic: Buy strong trends, avoid high volatility")
    print(f"     Best for: Bull markets")

    print(f"\n  2. MeanReversion:")
    print(f"     Factors: Bollinger, ZScore, Liquidity")
    print(f"     Logic: Buy oversold, sell overbought")
    print(f"     Best for: Ranging/Bear markets")

    print(f"\n  3. MultiFactor:")
    print(f"     Factors: Momentum, Bollinger, Volatility, Volume, Liquidity")
    print(f"     Logic: Combine trend + mean reversion")
    print(f"     Best for: All market conditions")

    print(f"\n  4. LongShortMulti:")
    print(f"     Factors: Same as MultiFactor")
    print(f"     Logic: Long strong, Short weak (market neutral)")
    print(f"     Best for: Volatile/uncertain markets")

    # 7. Generate HTML Report
    print("\n" + "=" * 70)
    print("GENERATING HTML REPORT")
    print("=" * 70)

    report_path = "quant_research_report.html"
    generator = HTMLReportGenerator(rolling_window=30, min_drawdown=0.01)
    generator.generate(
        result=best_result,
        output_path=report_path,
        title=f"Quantitative Research: {best_name} Strategy Analysis",
    )

    print(f"\nReport saved: {report_path}")
    print(f"\nThe report includes:")
    print(f"  - Equity curve visualization")
    print(f"  - Performance metrics summary")
    print(f"  - Performance attribution breakdown")
    print(f"  - Drawdown analysis with periods")
    print(f"  - Rolling 30-day Sharpe, Sortino, volatility")

    # 8. Final Summary
    print("\n" + "=" * 70)
    print("RESEARCH SUMMARY")
    print("=" * 70)

    print(f"\nBest Strategy (by Sharpe): {best_name}")
    print(f"Sharpe Ratio: {best_result.metrics.sharpe_ratio:.3f}")

    print(f"\nAll Strategies Performance:")
    for name, result in sorted(results.items(), key=lambda x: x[1].metrics.sharpe_ratio, reverse=True):
        print(f"  {name}: Return {result.metrics.total_return:>8.2%}, Sharpe {result.metrics.sharpe_ratio:>6.3f}")

    print(f"\nKey Findings:")
    print(f"  - Tested {len(strategies)} factor combinations")
    print(f"  - {len(best_result.trades)} trades executed (best strategy)")
    print(f"  - {mc_result.num_simulations} Monte Carlo simulations")
    print(f"  - Win rate: {best_result.metrics.win_rate:.1%}")
    print(f"  - Probability of loss: {mc_result.probability_of_loss:.1%}")

    print("\nDone")


if __name__ == "__main__":
    main()
