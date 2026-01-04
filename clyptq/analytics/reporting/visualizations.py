from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from clyptq.core.types import Fill, Snapshot


class PerformanceVisualizer:
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize

    def plot_equity_curve(
        self,
        snapshots: List[Snapshot],
        benchmark: Optional[pd.Series] = None,
        show_drawdown: bool = True,
    ) -> Figure:
        timestamps = [s.timestamp for s in snapshots]
        equity = [s.equity for s in snapshots]

        if show_drawdown:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, height_ratios=[2, 1])
        else:
            fig, ax1 = plt.subplots(figsize=self.figsize)

        ax1.plot(timestamps, equity, label="Strategy", linewidth=2)

        if benchmark is not None:
            ax1.plot(benchmark.index, benchmark.values, label="Benchmark", linewidth=2, alpha=0.7)

        ax1.set_xlabel("Date")
        ax1.set_ylabel("Equity")
        ax1.set_title("Equity Curve")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        if show_drawdown:
            equity_arr = np.array(equity)
            running_max = np.maximum.accumulate(equity_arr)
            drawdown = (equity_arr - running_max) / running_max

            ax2.fill_between(timestamps, drawdown, 0, alpha=0.3, color="red")
            ax2.plot(timestamps, drawdown, color="red", linewidth=1)
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Drawdown")
            ax2.set_title("Drawdown")
            ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

    def plot_returns_distribution(
        self, snapshots: List[Snapshot], bins: int = 50
    ) -> Figure:
        equity = [s.equity for s in snapshots]
        returns = pd.Series(equity).pct_change().dropna()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)

        ax1.hist(returns, bins=bins, alpha=0.7, edgecolor="black")
        ax1.axvline(returns.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {returns.mean():.4f}")
        ax1.axvline(returns.median(), color="green", linestyle="--", linewidth=2, label=f"Median: {returns.median():.4f}")
        ax1.set_xlabel("Return")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Returns Distribution")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        from scipy import stats
        stats.probplot(returns, dist="norm", plot=ax2)
        ax2.set_title("Q-Q Plot")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

    def plot_rolling_metrics(
        self,
        snapshots: List[Snapshot],
        window: int = 20,
        annualization: int = 252,
    ) -> Figure:
        timestamps = [s.timestamp for s in snapshots]
        equity = pd.Series([s.equity for s in snapshots], index=timestamps)
        returns = equity.pct_change().dropna()

        rolling_sharpe = (
            returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(annualization)
        )
        rolling_vol = returns.rolling(window).std() * np.sqrt(annualization)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, sharex=True)

        ax1.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2)
        ax1.axhline(0, color="red", linestyle="--", alpha=0.5)
        ax1.set_ylabel("Sharpe Ratio")
        ax1.set_title(f"Rolling Sharpe Ratio ({window}-period)")
        ax1.grid(True, alpha=0.3)

        ax2.plot(rolling_vol.index, rolling_vol.values, linewidth=2, color="orange")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Volatility")
        ax2.set_title(f"Rolling Volatility ({window}-period, annualized)")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

    def plot_monthly_returns(self, snapshots: List[Snapshot]) -> Figure:
        timestamps = [s.timestamp for s in snapshots]
        equity = pd.Series([s.equity for s in snapshots], index=timestamps)
        returns = equity.pct_change().dropna()

        monthly_returns = returns.resample("M").apply(lambda x: (1 + x).prod() - 1)

        years = monthly_returns.index.year.unique()
        months = range(1, 13)

        heatmap_data = np.full((len(years), 12), np.nan)

        for i, year in enumerate(years):
            year_data = monthly_returns[monthly_returns.index.year == year]
            for month_idx in year_data.index.month:
                heatmap_data[i, month_idx - 1] = year_data[year_data.index.month == month_idx].iloc[0]

        fig, ax = plt.subplots(figsize=self.figsize)

        im = ax.imshow(heatmap_data, cmap="RdYlGn", aspect="auto", vmin=-0.1, vmax=0.1)

        ax.set_xticks(range(12))
        ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
        ax.set_yticks(range(len(years)))
        ax.set_yticklabels(years)
        ax.set_title("Monthly Returns Heatmap")

        for i in range(len(years)):
            for j in range(12):
                if not np.isnan(heatmap_data[i, j]):
                    text = ax.text(j, i, f"{heatmap_data[i, j]:.1%}", ha="center", va="center", color="black", fontsize=8)

        fig.colorbar(im, ax=ax, label="Return")
        fig.tight_layout()
        return fig


class TradeVisualizer:
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize

    def plot_trade_pnl(self, fills: List[Fill]) -> Figure:
        if not fills:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, "No trades to display", ha="center", va="center", transform=ax.transAxes)
            return fig

        pnl_by_symbol = {}
        for fill in fills:
            if fill.symbol not in pnl_by_symbol:
                pnl_by_symbol[fill.symbol] = 0.0
            pnl = -fill.total_cost if fill.side.value == "BUY" else fill.total_cost
            pnl_by_symbol[fill.symbol] += pnl

        symbols = list(pnl_by_symbol.keys())
        pnls = list(pnl_by_symbol.values())

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)

        colors = ["green" if p > 0 else "red" for p in pnls]
        ax1.barh(symbols, pnls, color=colors, alpha=0.7)
        ax1.set_xlabel("P&L (USDT)")
        ax1.set_title("P&L by Symbol")
        ax1.grid(True, alpha=0.3, axis="x")

        ax2.hist(pnls, bins=20, alpha=0.7, edgecolor="black")
        ax2.axvline(0, color="red", linestyle="--", linewidth=2)
        ax2.set_xlabel("P&L (USDT)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("P&L Distribution")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

    def plot_trade_timeline(self, fills: List[Fill]) -> Figure:
        if not fills:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, "No trades to display", ha="center", va="center", transform=ax.transAxes)
            return fig

        buys = [f for f in fills if f.side.value == "BUY"]
        sells = [f for f in fills if f.side.value == "SELL"]

        fig, ax = plt.subplots(figsize=self.figsize)

        if buys:
            buy_times = [f.timestamp for f in buys]
            buy_prices = [f.price for f in buys]
            ax.scatter(buy_times, buy_prices, marker="^", color="green", s=100, alpha=0.6, label="Buy")

        if sells:
            sell_times = [f.timestamp for f in sells]
            sell_prices = [f.price for f in sells]
            ax.scatter(sell_times, sell_prices, marker="v", color="red", s=100, alpha=0.6, label="Sell")

        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.set_title("Trade Timeline")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig


class FactorVisualizer:
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize

    def plot_factor_correlation(
        self, factor_scores: Dict[str, Dict[str, float]], symbols: List[str]
    ) -> Figure:
        factor_names = list(factor_scores.keys())
        corr_matrix = np.zeros((len(factor_names), len(factor_names)))

        for i, f1 in enumerate(factor_names):
            for j, f2 in enumerate(factor_names):
                scores1 = [factor_scores[f1].get(s, np.nan) for s in symbols]
                scores2 = [factor_scores[f2].get(s, np.nan) for s in symbols]

                valid_idx = ~(np.isnan(scores1) | np.isnan(scores2))
                if valid_idx.sum() > 1:
                    corr_matrix[i, j] = np.corrcoef(
                        np.array(scores1)[valid_idx], np.array(scores2)[valid_idx]
                    )[0, 1]
                else:
                    corr_matrix[i, j] = np.nan

        fig, ax = plt.subplots(figsize=self.figsize)

        im = ax.imshow(corr_matrix, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)

        ax.set_xticks(range(len(factor_names)))
        ax.set_yticks(range(len(factor_names)))
        ax.set_xticklabels(factor_names, rotation=45, ha="right")
        ax.set_yticklabels(factor_names)
        ax.set_title("Factor Correlation Heatmap")

        for i in range(len(factor_names)):
            for j in range(len(factor_names)):
                if not np.isnan(corr_matrix[i, j]):
                    text = ax.text(j, i, f"{corr_matrix[i, j]:.2f}", ha="center", va="center", color="black")

        fig.colorbar(im, ax=ax, label="Correlation")
        fig.tight_layout()
        return fig

    def plot_factor_ic(
        self, ic_series: pd.Series, window: int = 20
    ) -> Figure:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, sharex=True)

        ax1.plot(ic_series.index, ic_series.values, linewidth=1, alpha=0.6)
        rolling_ic = ic_series.rolling(window).mean()
        ax1.plot(rolling_ic.index, rolling_ic.values, linewidth=2, label=f"{window}-period MA")
        ax1.axhline(0, color="red", linestyle="--", alpha=0.5)
        ax1.set_ylabel("IC")
        ax1.set_title("Information Coefficient Over Time")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        cumulative_ic = ic_series.cumsum()
        ax2.plot(cumulative_ic.index, cumulative_ic.values, linewidth=2, color="green")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Cumulative IC")
        ax2.set_title("Cumulative Information Coefficient")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

    def plot_quintile_returns(self, quintile_data: Dict) -> Figure:
        quintiles = [f"Q{i}" for i in range(1, 6)]
        mean_returns = [quintile_data[q]["mean_return"] for q in quintiles]
        std_returns = [quintile_data[q]["std_return"] for q in quintiles]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)

        colors = ["red", "orange", "yellow", "lightgreen", "green"]
        ax1.bar(quintiles, mean_returns, color=colors, alpha=0.7, edgecolor="black")
        ax1.set_xlabel("Quintile")
        ax1.set_ylabel("Mean Return")
        ax1.set_title("Mean Returns by Factor Quintile")
        ax1.grid(True, alpha=0.3, axis="y")

        sharpe_ratios = [
            quintile_data[q]["mean_return"] / quintile_data[q]["std_return"]
            if quintile_data[q]["std_return"] > 0
            else 0
            for q in quintiles
        ]
        ax2.bar(quintiles, sharpe_ratios, color=colors, alpha=0.7, edgecolor="black")
        ax2.set_xlabel("Quintile")
        ax2.set_ylabel("Sharpe Ratio")
        ax2.set_title("Sharpe Ratio by Factor Quintile")
        ax2.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        return fig


class PortfolioVisualizer:
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize

    def plot_portfolio_composition(
        self, snapshots: List[Snapshot], top_n: int = 10
    ) -> Figure:
        if not snapshots:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, "No snapshots to display", ha="center", va="center", transform=ax.transAxes)
            return fig

        timestamps = [s.timestamp for s in snapshots]
        all_symbols = set()
        for s in snapshots:
            all_symbols.update(s.positions.keys())

        symbol_weights = {symbol: [] for symbol in all_symbols}
        for snapshot in snapshots:
            total_value = sum(pos.market_value for pos in snapshot.positions.values())
            for symbol in all_symbols:
                if symbol in snapshot.positions and total_value > 0:
                    weight = snapshot.positions[symbol].market_value / total_value
                    symbol_weights[symbol].append(weight)
                else:
                    symbol_weights[symbol].append(0.0)

        avg_weights = {symbol: np.mean(weights) for symbol, weights in symbol_weights.items()}
        top_symbols = sorted(avg_weights.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_symbol_names = [s[0] for s in top_symbols]

        fig, ax = plt.subplots(figsize=self.figsize)

        bottom = np.zeros(len(timestamps))
        for symbol in top_symbol_names:
            weights = symbol_weights[symbol]
            ax.fill_between(range(len(timestamps)), bottom, bottom + weights, label=symbol, alpha=0.7)
            bottom += weights

        ax.set_xlabel("Time Period")
        ax.set_ylabel("Portfolio Weight")
        ax.set_title(f"Portfolio Composition Over Time (Top {top_n} Holdings)")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        return fig

    def plot_position_turnover(self, snapshots: List[Snapshot]) -> Figure:
        if len(snapshots) < 2:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, "Insufficient snapshots for turnover analysis", ha="center", va="center", transform=ax.transAxes)
            return fig

        timestamps = [s.timestamp for s in snapshots[1:]]
        turnovers = []

        for i in range(1, len(snapshots)):
            prev_symbols = set(snapshots[i - 1].positions.keys())
            curr_symbols = set(snapshots[i].positions.keys())

            added = len(curr_symbols - prev_symbols)
            removed = len(prev_symbols - curr_symbols)
            total_positions = max(len(prev_symbols), len(curr_symbols))

            if total_positions > 0:
                turnover = (added + removed) / (2 * total_positions)
            else:
                turnover = 0.0

            turnovers.append(turnover)

        fig, ax = plt.subplots(figsize=self.figsize)

        ax.plot(timestamps, turnovers, linewidth=2, marker="o", markersize=4)
        ax.axhline(np.mean(turnovers), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(turnovers):.2%}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Turnover Rate")
        ax.set_title("Portfolio Turnover Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

        fig.tight_layout()
        return fig
