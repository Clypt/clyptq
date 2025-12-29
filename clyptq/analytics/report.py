"""
HTML report generation.

Generate comprehensive backtest reports with metrics, attribution,
and visualizations.
"""

import base64
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt

from clyptq.analytics.attribution import PerformanceAttributor
from clyptq.analytics.drawdown import DrawdownAnalyzer
from clyptq.analytics.rolling import RollingMetricsCalculator
from clyptq.core.types import BacktestResult

matplotlib.use("Agg")


class HTMLReportGenerator:
    """
    Generate HTML backtest reports.

    Creates comprehensive reports with performance metrics,
    attribution analysis, and drawdown data.
    """

    def __init__(
        self,
        rolling_window: int = 30,
        min_drawdown: float = 0.01,
    ):
        """
        Initialize generator.

        Args:
            rolling_window: Window size for rolling metrics
            min_drawdown: Minimum drawdown to track
        """
        self.rolling_window = rolling_window
        self.min_drawdown = min_drawdown

    def generate(
        self,
        result: BacktestResult,
        output_path: str,
        title: Optional[str] = None,
    ) -> None:
        """
        Generate HTML report.

        Args:
            result: Backtest result
            output_path: Output file path
            title: Optional report title
        """
        if title is None:
            title = f"{result.strategy_name} Backtest Report"

        attributor = PerformanceAttributor()
        attribution = attributor.analyze(result)

        analyzer = DrawdownAnalyzer(min_drawdown=self.min_drawdown)
        drawdown = analyzer.analyze(result)

        rolling = None
        if len(result.snapshots) >= self.rolling_window:
            calculator = RollingMetricsCalculator(window=self.rolling_window)
            rolling = calculator.calculate(result)

        equity_chart = self._create_equity_chart(result)
        html = self._build_html(title, result, attribution, drawdown, rolling, equity_chart)

        Path(output_path).write_text(html, encoding="utf-8")

    def _create_equity_chart(self, result: BacktestResult) -> str:
        """Create equity curve chart as base64 string."""
        timestamps = [s.timestamp for s in result.snapshots]
        equity = [s.equity for s in result.snapshots]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(timestamps, equity, linewidth=2, color="#2563eb")
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity (USDT)")
        ax.set_title("Equity Curve", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style="plain", axis="y")

        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)

        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        return f"data:image/png;base64,{img_base64}"

    def _build_html(
        self,
        title: str,
        result: BacktestResult,
        attribution,
        drawdown,
        rolling,
        equity_chart: str,
    ) -> str:
        """Build HTML content."""
        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        {self._get_css()}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        {self._build_summary(result)}
        {self._build_equity_section(equity_chart)}
        {self._build_metrics(result)}
        {self._build_attribution(attribution)}
        {self._build_drawdown(drawdown)}
        {self._build_rolling(rolling) if rolling else ''}
    </div>
</body>
</html>"""

    def _get_css(self) -> str:
        """Get CSS styles."""
        return """
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #f5f5f5;
            margin: 0;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            margin-bottom: 10px;
        }
        h2 {
            color: #666;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
            margin-top: 40px;
        }
        .timestamp {
            color: #999;
            margin-bottom: 30px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th {
            background: #f8f8f8;
            text-align: left;
            padding: 12px;
            border-bottom: 2px solid #ddd;
        }
        td {
            padding: 10px 12px;
            border-bottom: 1px solid #eee;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .metric-card {
            background: #f8f8f8;
            padding: 20px;
            border-radius: 6px;
        }
        .metric-label {
            color: #666;
            font-size: 14px;
            margin-bottom: 5px;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }
        .positive { color: #22c55e; }
        .negative { color: #ef4444; }
        """

    def _build_summary(self, result: BacktestResult) -> str:
        """Build summary section."""
        m = result.metrics
        return f"""
        <h2>Summary</h2>
        <table>
            <tr><th>Strategy</th><td>{result.strategy_name}</td></tr>
            <tr><th>Mode</th><td>{result.mode}</td></tr>
            <tr><th>Period</th><td>{m.start_date.date()} to {m.end_date.date()}</td></tr>
            <tr><th>Duration</th><td>{m.duration_days} days</td></tr>
            <tr><th>Trades</th><td>{m.num_trades}</td></tr>
        </table>
        """

    def _build_equity_section(self, equity_chart: str) -> str:
        """Build equity curve section."""
        return f"""
        <h2>Equity Curve</h2>
        <div style="text-align: center; margin: 20px 0;">
            <img src="{equity_chart}" style="max-width: 100%; height: auto;">
        </div>
        """

    def _build_metrics(self, result: BacktestResult) -> str:
        """Build metrics section."""
        m = result.metrics
        ret_class = "positive" if m.total_return > 0 else "negative"
        dd_class = "negative"

        return f"""
        <h2>Performance Metrics</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Total Return</div>
                <div class="metric-value {ret_class}">{m.total_return:.2%}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Annualized Return</div>
                <div class="metric-value {ret_class}">{m.annualized_return:.2%}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-value">{m.sharpe_ratio:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Sortino Ratio</div>
                <div class="metric-value">{m.sortino_ratio:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Volatility</div>
                <div class="metric-value">{m.volatility:.2%}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value {dd_class}">{m.max_drawdown:.2%}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Win Rate</div>
                <div class="metric-value">{m.win_rate:.2%}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Profit Factor</div>
                <div class="metric-value">{m.profit_factor:.2f}</div>
            </div>
        </div>
        """

    def _build_attribution(self, attribution) -> str:
        """Build attribution section."""
        html = f"""
        <h2>Performance Attribution</h2>
        <table>
            <tr><th>Component</th><th>Value</th></tr>
            <tr><td>Total Return</td><td class="{'positive' if attribution.total_return > 0 else 'negative'}">{attribution.total_return:.2%}</td></tr>
            <tr><td>Transaction Costs</td><td class="negative">{attribution.transaction_cost_drag:.2%}</td></tr>
            <tr><td>Cash Drag</td><td class="negative">{attribution.cash_drag:.2%}</td></tr>
        </table>
        """

        if attribution.asset_attributions:
            html += """
            <h3>Top Asset Contributors</h3>
            <table>
                <tr><th>Asset</th><th>Weight Contribution</th></tr>
            """
            for asset in attribution.asset_attributions[:10]:
                html += f"""
                <tr>
                    <td>{asset.symbol}</td>
                    <td>{asset.weight_contribution:.2%}</td>
                </tr>
                """
            html += "</table>"

        return html

    def _build_drawdown(self, drawdown) -> str:
        """Build drawdown section."""
        html = f"""
        <h2>Drawdown Analysis</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Max Drawdown</td><td class="negative">{drawdown.max_drawdown:.2%}</td></tr>
            <tr><td>Avg Drawdown</td><td class="negative">{drawdown.avg_drawdown:.2%}</td></tr>
            <tr><td>Drawdown Periods</td><td>{len(drawdown.drawdown_periods)}</td></tr>
        </table>
        """

        if drawdown.drawdown_periods:
            html += """
            <h3>Top 5 Drawdowns</h3>
            <table>
                <tr><th>Start</th><th>End</th><th>Depth</th><th>Duration</th><th>Recovery</th></tr>
            """
            for period in drawdown.drawdown_periods[:5]:
                recovery = (
                    f"{period.recovery_days} days"
                    if period.recovery_days
                    else "Not recovered"
                )
                html += f"""
                <tr>
                    <td>{period.start.date()}</td>
                    <td>{period.end.date()}</td>
                    <td class="negative">{period.depth:.2%}</td>
                    <td>{period.duration_days} days</td>
                    <td>{recovery}</td>
                </tr>
                """
            html += "</table>"

        return html

    def _build_rolling(self, rolling) -> str:
        """Build rolling metrics section."""
        avg_sharpe = sum(rolling.sharpe_ratio) / len(rolling.sharpe_ratio)
        avg_vol = sum(rolling.volatility) / len(rolling.volatility)

        return f"""
        <h2>Rolling Metrics ({self.rolling_window}-day window)</h2>
        <table>
            <tr><th>Metric</th><th>Average</th></tr>
            <tr><td>Sharpe Ratio</td><td>{avg_sharpe:.2f}</td></tr>
            <tr><td>Volatility</td><td>{avg_vol:.2%}</td></tr>
        </table>
        """
