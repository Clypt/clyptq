"""
Monte Carlo simulation for strategy validation and risk assessment.

Bootstrap sampling from historical returns to generate confidence intervals
and risk metrics through repeated random sampling.
"""

import numpy as np
from typing import List, Optional
from datetime import datetime

from clyptq.core.types import BacktestResult, MonteCarloResult


class MonteCarloSimulator:
    """
    Monte Carlo simulator for backtesting results.

    Uses bootstrap resampling of daily returns to generate distribution of
    possible outcomes. Calculates confidence intervals, risk metrics, and
    probability distributions.
    """

    def __init__(
        self,
        num_simulations: int = 1000,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize Monte Carlo simulator.

        Args:
            num_simulations: Number of simulation runs
            random_seed: Random seed for reproducibility
        """
        self.num_simulations = num_simulations
        self.random_seed = random_seed

        if random_seed is not None:
            np.random.seed(random_seed)

    def run(
        self,
        backtest_result: BacktestResult,
        initial_capital: Optional[float] = None,
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation on backtest results.

        Args:
            backtest_result: Historical backtest results
            initial_capital: Starting capital (from backtest if not provided)

        Returns:
            MonteCarloResult with distribution statistics
        """
        # Extract daily returns from snapshots
        daily_returns = self._extract_daily_returns(backtest_result)

        if len(daily_returns) < 2:
            raise ValueError("Insufficient data for Monte Carlo simulation")

        if initial_capital is None:
            initial_capital = backtest_result.snapshots[0].equity

        # Run simulations
        final_equities = []
        equity_paths = []
        sharpe_ratios = []

        for _ in range(self.num_simulations):
            # Bootstrap sample returns (with replacement)
            sampled_returns = np.random.choice(
                daily_returns, size=len(daily_returns), replace=True
            )

            # Generate equity path
            equity_path = self._simulate_equity_path(
                initial_capital, sampled_returns
            )

            equity_paths.append(equity_path)
            final_equities.append(equity_path[-1])

            # Calculate Sharpe for this path
            sharpe = self._calculate_sharpe(sampled_returns)
            sharpe_ratios.append(sharpe)

        # Calculate statistics
        return self._calculate_statistics(
            final_equities=final_equities,
            equity_paths=equity_paths,
            sharpe_ratios=sharpe_ratios,
            initial_capital=initial_capital,
            simulation_days=len(daily_returns),
        )

    def _extract_daily_returns(self, result: BacktestResult) -> np.ndarray:
        """Extract daily returns from backtest snapshots."""
        equities = [snap.equity for snap in result.snapshots]

        if len(equities) < 2:
            return np.array([])

        returns = np.diff(equities) / equities[:-1]
        return returns

    def _simulate_equity_path(
        self, initial_capital: float, returns: np.ndarray
    ) -> List[float]:
        """Simulate equity path from returns."""
        equity = initial_capital
        path = [equity]

        for ret in returns:
            equity = equity * (1 + ret)
            path.append(equity)

        return path

    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio from returns."""
        if len(returns) == 0:
            return 0.0

        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)

        if std_return == 0:
            return 0.0

        # Annualized Sharpe (assuming daily returns)
        sharpe = (mean_return / std_return) * np.sqrt(252)
        return sharpe

    def _calculate_statistics(
        self,
        final_equities: List[float],
        equity_paths: List[List[float]],
        sharpe_ratios: List[float],
        initial_capital: float,
        simulation_days: int,
    ) -> MonteCarloResult:
        """Calculate summary statistics from simulation results."""
        final_equities = np.array(final_equities)
        sharpe_ratios = np.array(sharpe_ratios)

        # Returns
        returns = (final_equities / initial_capital) - 1.0

        # Summary statistics
        mean_return = np.mean(returns)
        median_return = np.median(returns)
        std_return = np.std(returns, ddof=1)

        # Confidence intervals
        ci_5_return = np.percentile(returns, 5)
        ci_50_return = np.percentile(returns, 50)
        ci_95_return = np.percentile(returns, 95)

        # Risk metrics
        probability_of_loss = np.mean(returns < 0)

        # Expected Shortfall (CVaR) at 5%
        worst_5_percent = np.sort(returns)[: int(0.05 * len(returns))]
        expected_shortfall_5 = (
            np.mean(worst_5_percent) if len(worst_5_percent) > 0 else 0.0
        )

        # Max drawdown percentiles
        max_drawdowns = [self._calculate_max_drawdown(path) for path in equity_paths]
        max_drawdown_5 = np.percentile(max_drawdowns, 5)
        max_drawdown_50 = np.percentile(max_drawdowns, 50)
        max_drawdown_95 = np.percentile(max_drawdowns, 95)

        # Sharpe statistics
        mean_sharpe = np.mean(sharpe_ratios)
        ci_5_sharpe = np.percentile(sharpe_ratios, 5)
        ci_95_sharpe = np.percentile(sharpe_ratios, 95)

        return MonteCarloResult(
            num_simulations=self.num_simulations,
            final_equities=final_equities.tolist(),
            equity_paths=equity_paths,
            mean_return=mean_return,
            median_return=median_return,
            std_return=std_return,
            ci_5_return=ci_5_return,
            ci_50_return=ci_50_return,
            ci_95_return=ci_95_return,
            probability_of_loss=probability_of_loss,
            expected_shortfall_5=expected_shortfall_5,
            max_drawdown_5=max_drawdown_5,
            max_drawdown_50=max_drawdown_50,
            max_drawdown_95=max_drawdown_95,
            mean_sharpe=mean_sharpe,
            ci_5_sharpe=ci_5_sharpe,
            ci_95_sharpe=ci_95_sharpe,
            initial_capital=initial_capital,
            simulation_days=simulation_days,
            timestamp=datetime.now(),
        )

    def _calculate_max_drawdown(self, equity_path: List[float]) -> float:
        """Calculate maximum drawdown from equity path."""
        if len(equity_path) < 2:
            return 0.0

        equity_arr = np.array(equity_path)
        running_max = np.maximum.accumulate(equity_arr)
        drawdown = (equity_arr - running_max) / running_max

        return abs(np.min(drawdown))


def print_monte_carlo_results(result: MonteCarloResult) -> None:
    """Print Monte Carlo simulation results in readable format."""
    print("=" * 70)
    print("MONTE CARLO SIMULATION RESULTS")
    print("=" * 70)

    print(f"\nSimulation Parameters:")
    print(f"  Number of Simulations: {result.num_simulations:,}")
    print(f"  Simulation Days:       {result.simulation_days}")
    print(f"  Initial Capital:       ${result.initial_capital:,.2f}")

    print(f"\nReturn Distribution:")
    print(f"  Mean Return:           {result.mean_return:>10.2%}")
    print(f"  Median Return:         {result.median_return:>10.2%}")
    print(f"  Std Deviation:         {result.std_return:>10.2%}")

    print(f"\nConfidence Intervals:")
    print(f"  5th Percentile:        {result.ci_5_return:>10.2%}")
    print(f"  50th Percentile:       {result.ci_50_return:>10.2%}")
    print(f"  95th Percentile:       {result.ci_95_return:>10.2%}")

    print(f"\nRisk Metrics:")
    print(f"  Probability of Loss:   {result.probability_of_loss:>10.2%}")
    print(f"  Expected Shortfall (5%):{result.expected_shortfall_5:>10.2%}")
    print(f"  Max DD (5th %ile):     {result.max_drawdown_5:>10.2%}")
    print(f"  Max DD (50th %ile):    {result.max_drawdown_50:>10.2%}")
    print(f"  Max DD (95th %ile):    {result.max_drawdown_95:>10.2%}")

    print(f"\nSharpe Ratio Distribution:")
    print(f"  Mean Sharpe:           {result.mean_sharpe:>10.2f}")
    print(f"  5th Percentile:        {result.ci_5_sharpe:>10.2f}")
    print(f"  95th Percentile:       {result.ci_95_sharpe:>10.2f}")

    print("\n" + "=" * 70)
