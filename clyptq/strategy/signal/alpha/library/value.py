"""Value-based alpha signals."""

from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class RealizedSpreadAlpha(BaseSignal):
    """Realized spread alpha: lower spread = higher score.

    Spread proxy = (high - low) / close
    """

    role = SignalRole.ALPHA
    default_params = {"lookback": 20}

    def compute(self, data):
        """Compute realized spread scores.

        Args:
            data: Dict or UniverseData with 'high', 'low', 'close' DataFrames

        Returns:
            DataFrame of spread scores (higher = tighter spread)
        """
        high = data['high'] if isinstance(data, dict) else data.high
        low = data['low'] if isinstance(data, dict) else data.low
        close = data['close'] if isinstance(data, dict) else data.close
        lookback = self.params.get("lookback", self.lookback)

        # Spread = (high - low) / close
        range_val = operator.sub(high, low)
        spread = operator.div(range_val, close)

        # Average spread
        avg_spread = operator.ts_mean(spread, lookback)

        # Negative: prefer low spread
        score = operator.mul(avg_spread, -1)

        return operator.ts_fillna(score, 0)


class PriceEfficiencyAlpha(BaseSignal):
    """Price efficiency alpha: how close price is to mid of range.

    Lower deviation from mid = higher efficiency.
    """

    role = SignalRole.ALPHA
    default_params = {"lookback": 20}

    def compute(self, data):
        """Compute price efficiency scores.

        Args:
            data: Dict or UniverseData with 'high', 'low', 'close' DataFrames

        Returns:
            DataFrame of efficiency scores
        """
        high = data['high'] if isinstance(data, dict) else data.high
        low = data['low'] if isinstance(data, dict) else data.low
        close = data['close'] if isinstance(data, dict) else data.close
        lookback = self.params.get("lookback", self.lookback)

        # Mid price = (high + low) / 2
        mid_price = operator.div(operator.add(high, low), 2.0)

        # Range
        price_range = operator.sub(high, low)

        # Deviation from mid
        deviation = operator.abs(operator.sub(close, mid_price))
        relative_deviation = operator.div(deviation, price_range)

        # Average deviation
        avg_deviation = operator.ts_mean(relative_deviation, lookback)

        # Negative: prefer price close to mid
        score = operator.mul(avg_deviation, -1)

        return operator.ts_fillna(score, 0)


class ImpliedBasisAlpha(BaseSignal):
    """Implied basis alpha: momentum / volatility (risk-adjusted momentum).

    Also known as Sharpe ratio or information ratio proxy.
    """

    role = SignalRole.ALPHA
    default_params = {"lookback": 20}

    def compute(self, data):
        """Compute implied basis (risk-adjusted momentum) scores.

        Args:
            data: Dict or UniverseData with 'close' DataFrame

        Returns:
            DataFrame of risk-adjusted momentum scores
        """
        close = data['close'] if isinstance(data, dict) else data.close
        lookback = self.params.get("lookback", self.lookback)

        # Calculate returns
        returns = operator.ts_returns(close, 1)

        # Mean and std of returns
        mean_return = operator.ts_mean(returns, lookback)
        std_return = operator.ts_std(returns, lookback)

        # Sharpe-like ratio = mean / std
        basis_proxy = operator.div(mean_return, std_return)

        return operator.ts_fillna(basis_proxy, 0)
