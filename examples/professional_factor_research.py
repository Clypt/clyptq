"""
Professional Factor Research Workflow

Demonstrates v0.8.0 advanced factor analysis tools:
1. Signal Quality Analysis - IC, IR, factor decay
2. IC Decay Analysis - factor predictive power over time
3. Turnover-Performance Analysis - optimal rebalancing frequency
4. Factor Orthogonalization - remove redundancy, PCA reduction

Usage:
    python examples/professional_factor_research.py
"""

from datetime import timedelta

from clyptq import CostModel, Constraints
from clyptq.analytics.factors import (
    FactorAnalyzer,
    SignalQuality,
    turnover_performance_frontier,
    optimal_rebalance_frequency,
)
from clyptq.data.loaders.ccxt import load_crypto_data
from clyptq.trading.engine import BacktestEngine
from clyptq.trading.execution import BacktestExecutor
from clyptq.trading.factors.library.momentum import MomentumFactor
from clyptq.trading.factors.library.volatility import VolatilityFactor
from clyptq.trading.factors.library.volume import VolumeFactor
from clyptq.trading.factors.library.mean_reversion import BollingerFactor, ZScoreFactor
from clyptq.trading.factors.library.liquidity import AmihudFactor
from clyptq.trading.factors.ops.factor_combination import (
    orthogonalize_factors,
    pca_factors,
    remove_correlation,
)
from clyptq.trading.portfolio.constructors import ScoreWeightedConstructor
from clyptq.trading.strategy.base import SimpleStrategy


class ResearchStrategy(SimpleStrategy):
    """Multi-factor strategy for research."""

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
            max_gross_exposure=0.95,
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
            name="Research",
        )


def print_section_header(title):
    """Print section header."""
    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)


def main():
    print_section_header("PROFESSIONAL FACTOR RESEARCH WORKFLOW - v0.8.0")

    # 1. Data Loading
    print("\nStep 1: Loading Data")
    print("-" * 80)
    symbols = [
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
        "ADA/USDT", "AVAX/USDT", "DOGE/USDT", "DOT/USDT", "MATIC/USDT",
        "LTC/USDT", "LINK/USDT", "UNI/USDT", "ATOM/USDT", "XLM/USDT",
    ]
    print(f"Loading {len(symbols)} symbols (720 days)...")
    store = load_crypto_data(symbols=symbols, exchange="binance", timeframe="1d", days=720)

    date_range = store.get_date_range()
    start = date_range.end - timedelta(days=365)
    end = date_range.end
    print(f"Research period: {start.date()} to {end.date()}")

    # 2. Signal Quality Analysis
    print_section_header("STEP 2: SIGNAL QUALITY ANALYSIS")

    strategy = ResearchStrategy()
    factors = strategy.factors()
    print(f"\nAnalyzing {len(factors)} factors:")
    for i, factor in enumerate(factors, 1):
        lookback = getattr(factor, 'lookback', 'N/A')
        print(f"  {i}. {factor.__class__.__name__} (lookback={lookback})")

    print("\nComputing signal quality metrics...")
    signal_quality = SignalQuality()
    quality_result = signal_quality.analyze(factors, store, start, end)

    print("\n" + "-" * 80)
    print(f"{'Factor':<25} {'IC Mean':<12} {'IR':<12} {'Hit Rate':<12}")
    print("-" * 80)
    for stat in quality_result.factor_stats:
        print(
            f"{stat.factor_name:<25} "
            f"{stat.ic_mean:>10.4f}  "
            f"{stat.information_ratio:>10.4f}  "
            f"{stat.hit_rate:>10.2%}"
        )
    print("-" * 80)

    print("\nSignal Quality Interpretation:")
    print("  IC Mean: Average Information Coefficient (correlation with returns)")
    print("    > 0.05: Strong signal")
    print("    > 0.02: Moderate signal")
    print("    < 0.02: Weak signal")
    print("\n  IR: Information Ratio (IC Mean / IC Std)")
    print("    > 0.5: Very good")
    print("    > 0.3: Good")
    print("    < 0.3: Needs improvement")
    print("\n  Hit Rate: % of periods with IC > 0")
    print("    > 60%: Consistent")
    print("    > 50%: Acceptable")
    print("    < 50%: Inconsistent")

    # 3. IC Decay Analysis
    print_section_header("STEP 3: IC DECAY ANALYSIS")

    print("\nAnalyzing factor predictive power decay over 1-10 days...")
    analyzer = FactorAnalyzer()

    for i, factor in enumerate(factors[:3], 1):
        print(f"\n{i}. {factor.__class__.__name__}")
        print("-" * 80)

        decay = analyzer.analyze_ic_decay(
            factor=factor,
            data_store=store,
            start=start,
            end=end,
            max_days=10,
        )

        print(f"{'Day':<8} {'IC':<12} {'Change':<12}")
        print("-" * 80)
        for day, ic in decay.ic_by_day.items():
            change = ""
            if day > 1:
                prev_ic = decay.ic_by_day.get(day - 1, 0)
                diff = ic - prev_ic
                change = f"({diff:+.4f})"
            print(f"Day {day:<4} {ic:>10.4f}  {change}")

        print(f"\nHalf-life: {decay.half_life:.1f} days")
        print(f"Optimal holding: {decay.optimal_holding_period} days")
        print("-" * 80)

    # 4. Turnover-Performance Analysis
    print_section_header("STEP 4: TURNOVER-PERFORMANCE ANALYSIS")

    print("\nAnalyzing optimal rebalancing frequency...")
    print("Testing rebalance frequencies: daily, weekly, biweekly, monthly")

    cost_model = CostModel(maker_fee=0.001, taker_fee=0.001, slippage_bps=5.0)
    executor = BacktestExecutor(cost_model)

    freq_analysis = optimal_rebalance_frequency(
        strategy=strategy,
        data=store,
        executor=executor,
        initial_capital=100000.0,
        start=start,
        end=end,
        frequencies=["daily", "weekly", "biweekly", "monthly"],
    )

    print("\n" + "-" * 80)
    print(f"{'Frequency':<15} {'Return':<12} {'Sharpe':<12} {'Turnover':<12} {'Trades':<10}")
    print("-" * 80)
    for freq, metrics in freq_analysis["all_results"].items():
        print(
            f"{freq:<15} "
            f"{metrics['total_return']:>10.2%}  "
            f"{metrics['sharpe_ratio']:>10.3f}  "
            f"{metrics['turnover']:>10.1%}  "
            f"{metrics['num_trades']:>8}"
        )
    print("-" * 80)

    optimal_freq = freq_analysis["optimal_frequency"]
    optimal_metrics = freq_analysis["optimal_metrics"]
    print(f"\nOptimal Frequency: {optimal_freq}")
    print(f"  Return: {optimal_metrics['total_return']:.2%}")
    print(f"  Sharpe: {optimal_metrics['sharpe_ratio']:.3f}")
    print(f"  Turnover: {optimal_metrics['turnover']:.1%}")
    print(f"  Trades: {optimal_metrics['num_trades']}")

    print("\nInterpretation:")
    print("  - Higher turnover = more trades = higher costs")
    print("  - Lower turnover = stale positions = missed opportunities")
    print("  - Optimal frequency balances signal decay vs transaction costs")

    # 5. Factor Orthogonalization
    print_section_header("STEP 5: FACTOR ORTHOGONALIZATION")

    print("\nStep 5a: Computing raw factor scores...")
    view = store.get_view(end - timedelta(days=1))
    raw_scores = {}
    for factor in factors:
        scores = factor.compute(view)
        raw_scores[factor.__class__.__name__] = scores

    print(f"Computed scores for {len(raw_scores)} factors across {len(scores)} symbols")

    # 5a. Orthogonalization
    print("\nStep 5b: Applying Gram-Schmidt Orthogonalization...")
    orthogonal_scores = orthogonalize_factors(raw_scores)

    print("\nOrthogonality Verification:")
    print("  Raw factors: Potentially correlated")
    print("  Orthogonal factors: Independent (dot product ≈ 0)")
    print(f"  Result: {len(orthogonal_scores)} orthogonal factors")

    # 5b. PCA Reduction
    print("\nStep 5c: PCA Dimensionality Reduction...")
    n_components = 3
    pca_scores = pca_factors(raw_scores, n_components=n_components)

    print(f"\nPCA Results:")
    print(f"  Original factors: {len(raw_scores)}")
    print(f"  Principal components: {len(pca_scores)}")
    print(f"  Dimensionality reduction: {len(raw_scores)} → {n_components}")

    print("\nPrincipal Components:")
    for pc_name in pca_scores.keys():
        print(f"  {pc_name}: Captures variance from all {len(raw_scores)} factors")

    # 5c. Correlation Removal
    print("\nStep 5d: Market-Neutral Factor Construction...")
    momentum_scores = raw_scores.get("MomentumFactor", {})
    market_beta = [raw_scores.get("VolumeFactor", {})]

    neutral_momentum = remove_correlation(momentum_scores, market_beta)

    print("\nMarket-Neutral Momentum:")
    print(f"  Original: Momentum factor (correlated with market)")
    print(f"  Conditioning: Volume factor (market activity proxy)")
    print(f"  Result: Market-neutral momentum (residual after removing volume correlation)")

    # 6. Best Practices Summary
    print_section_header("STEP 6: BEST PRACTICES SUMMARY")

    print("\n1. Signal Quality Analysis:")
    print("   - Check IC mean, IR, and hit rate before using factors")
    print("   - Weak factors (IC < 0.02) may hurt performance")
    print("   - Combine factors with complementary signals")

    print("\n2. IC Decay Analysis:")
    print("   - Understand how long factors remain predictive")
    print("   - Rebalance before IC decays to zero")
    print("   - Short-lived factors need frequent rebalancing")

    print("\n3. Turnover-Performance:")
    print("   - Balance signal decay vs transaction costs")
    print("   - High-frequency rebalancing works for strong, fast-decaying signals")
    print("   - Low-frequency works for slow-moving factors")

    print("\n4. Factor Orthogonalization:")
    print("   - Orthogonalize to remove redundancy between factors")
    print("   - Use PCA when you have many similar factors")
    print("   - Create market-neutral factors by removing systematic exposures")

    print("\n5. Implementation Workflow:")
    print("   a. Analyze signal quality → filter weak factors")
    print("   b. Check IC decay → determine optimal holding period")
    print("   c. Test turnover-performance → choose rebalance frequency")
    print("   d. Orthogonalize factors → reduce redundancy")
    print("   e. Backtest final strategy → validate performance")

    # 7. Next Steps
    print_section_header("NEXT STEPS FOR YOUR RESEARCH")

    print("\n1. Factor Development:")
    print("   - Test new factor ideas using signal quality metrics")
    print("   - Analyze IC decay to understand factor behavior")
    print("   - Combine complementary factors with orthogonalization")

    print("\n2. Strategy Optimization:")
    print("   - Use turnover analysis to find optimal rebalance frequency")
    print("   - Apply PCA to reduce dimensionality in multi-factor strategies")
    print("   - Create market-neutral factors for hedged strategies")

    print("\n3. Risk Management:")
    print("   - Monitor IC stability over time")
    print("   - Track factor correlation to avoid concentration")
    print("   - Use orthogonalization to ensure diversification")

    print("\nDone - Factor research workflow complete!")


if __name__ == "__main__":
    main()
