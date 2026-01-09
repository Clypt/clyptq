"""Momentum-based alpha signals."""

from typing import Dict, List, Optional

from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class MomentumAlpha(BaseSignal):
    """Simple momentum alpha: return over lookback period.

    Example:
        ```python
        alpha = MomentumAlpha(lookback=20)
        result = alpha.compute(data)  # Returns DataFrame
        ```
    """

    role = SignalRole.ALPHA
    default_params = {"lookback": 20}

    def compute(self, data):
        """Compute momentum scores.

        Args:
            data: Dict or UniverseData with 'close' DataFrame (timestamps x symbols)

        Returns:
            DataFrame of momentum scores (timestamps x symbols)
        """
        close = data['close'] if isinstance(data, dict) else data.close
        lookback = self.params.get("lookback", self.lookback)

        # Return = (current - past) / past
        past_close = operator.ts_delay(close, lookback)
        momentum = operator.div(
            operator.sub(close, past_close),
            past_close
        )

        return operator.ts_fillna(momentum, 0)


class RSIAlpha(BaseSignal):
    """RSI alpha normalized to [-1, 1].

    RSI > 50 gives positive scores (bullish)
    RSI < 50 gives negative scores (bearish)
    """

    role = SignalRole.ALPHA
    default_params = {"lookback": 14}

    def compute(self, data):
        """Compute RSI-based alpha scores.

        Args:
            data: Dict or UniverseData with 'close' DataFrame (timestamps x symbols)

        Returns:
            DataFrame of RSI scores in range [-1, 1]
        """
        close = data['close'] if isinstance(data, dict) else data.close
        lookback = self.params.get("lookback", self.lookback)

        # Calculate price changes
        delta = operator.ts_delta(close, 1)

        # Separate gains and losses
        gains = operator.clip(delta, lower=0)
        losses = operator.clip(operator.neg(delta), lower=0)

        # Calculate average gains and losses
        avg_gain = operator.ts_mean(gains, lookback)
        avg_loss = operator.ts_mean(losses, lookback)

        # Calculate RSI
        rs = operator.div(avg_gain, avg_loss)
        rsi = operator.sub(100, operator.div(100, operator.add(1, rs)))

        # Normalize to [-1, 1]
        normalized = operator.div(operator.sub(rsi, 50), 50)

        return operator.ts_fillna(normalized, 0)


class TrendStrengthAlpha(BaseSignal):
    """Trend strength alpha using linear regression slope.

    Computes annualized slope of log prices as a measure of trend strength.
    """

    role = SignalRole.ALPHA
    default_params = {"lookback": 20}

    def compute(self, data):
        """Compute trend strength scores.

        Args:
            data: Dict or UniverseData with 'close' DataFrame (timestamps x symbols)

        Returns:
            DataFrame of trend strength scores (annualized log slope)
        """
        close = data['close'] if isinstance(data, dict) else data.close
        lookback = self.params.get("lookback", self.lookback)

        # Use log prices for slope calculation
        log_close = operator.log(close)

        # Calculate slope using ts_slope
        slope = operator.ts_slope(log_close, lookback)

        # Annualize (assuming daily data)
        annualized = operator.mul(slope, 252)

        return operator.ts_fillna(annualized, 0)


class MultiTimeframeMomentumAlpha(BaseSignal):
    """Multi-timeframe momentum alpha.

    Combines momentum signals from multiple timeframes with weighted average.
    Default implementation uses single-timeframe momentum.
    """

    role = SignalRole.ALPHA
    default_params = {"lookback": 20}

    def __init__(
        self,
        timeframes: List[str] = None,
        lookbacks: Optional[Dict[str, int]] = None,
        weights: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        # Default to single timeframe if not specified
        if timeframes is None:
            timeframes = ["1d"]

        self.timeframes = timeframes
        self.lookbacks = lookbacks or {tf: 20 for tf in timeframes}

        if weights is None:
            weight = 1.0 / len(timeframes)
            self.weights = {tf: weight for tf in timeframes}
        else:
            self.weights = weights

        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")

        max_lookback = max(self.lookbacks.values())

        # Initialize with max lookback
        super().__init__(lookback=max_lookback, **kwargs)

    def compute(self, data):
        """Compute weighted momentum across timeframes.

        Note: For multi-timeframe support, use ResamplingDataStore.get_view_at_timeframe()
        to get views at different timeframes and combine them externally.

        This implementation uses single-timeframe momentum as a fallback.
        """
        close = data['close'] if isinstance(data, dict) else data.close
        lookback = self.params.get("lookback", self.lookback)

        # Calculate simple momentum
        past_close = operator.ts_delay(close, lookback)
        momentum = operator.div(
            operator.sub(close, past_close),
            past_close
        )

        return operator.ts_fillna(momentum, 0)
