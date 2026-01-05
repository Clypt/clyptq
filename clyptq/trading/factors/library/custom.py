from clyptq.core.base import Factor
from clyptq.data.stores.store import DataView
from clyptq.trading.factors.ops.time_series import ts_mean, ts_std, ts_max, ts_min
import numpy as np


class RSIFactor(Factor):
    def __init__(self, lookback: int = 14):
        self.lookback = lookback

    def compute(self, view: DataView) -> dict[str, float]:
        scores = {}
        for symbol in view.symbols:
            prices = view.close(symbol, self.lookback + 1)
            if len(prices) < self.lookback + 1:
                continue

            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)

            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)

            if avg_loss == 0:
                rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))

            scores[symbol] = (rsi - 50) / 50

        return scores


class MACDFactor(Factor):
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def compute(self, view: DataView) -> dict[str, float]:
        scores = {}
        lookback = self.slow + self.signal

        for symbol in view.symbols:
            prices = view.close(symbol, lookback)
            if len(prices) < lookback:
                continue

            ema_fast = self._ema(prices, self.fast)
            ema_slow = self._ema(prices, self.slow)
            macd_line = ema_fast - ema_slow

            macd_values = []
            for i in range(len(prices) - self.slow + 1):
                p = prices[i : i + self.slow]
                ef = self._ema(p, self.fast)
                es = self._ema(p, self.slow)
                macd_values.append(ef - es)

            if len(macd_values) < self.signal:
                continue

            signal_line = self._ema(np.array(macd_values), self.signal)
            macd_hist = macd_line - signal_line

            scores[symbol] = macd_hist / (abs(macd_line) + 1e-10)

        return scores

    def _ema(self, prices: np.ndarray, period: int) -> float:
        multiplier = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        return ema


class ATRFactor(Factor):
    def __init__(self, lookback: int = 14):
        self.lookback = lookback

    def compute(self, view: DataView) -> dict[str, float]:
        scores = {}

        for symbol in view.symbols:
            highs = view.high(symbol, self.lookback + 1)
            lows = view.low(symbol, self.lookback + 1)
            closes = view.close(symbol, self.lookback + 1)

            if len(highs) < self.lookback + 1:
                continue

            true_ranges = []
            for i in range(1, len(highs)):
                high_low = highs[i] - lows[i]
                high_close = abs(highs[i] - closes[i - 1])
                low_close = abs(lows[i] - closes[i - 1])
                true_ranges.append(max(high_low, high_close, low_close))

            atr = np.mean(true_ranges)
            current_price = closes[-1]

            scores[symbol] = -atr / current_price

        return scores


class VolumeSpikeFactor(Factor):
    def __init__(self, lookback: int = 20, spike_threshold: float = 2.0):
        self.lookback = lookback
        self.spike_threshold = spike_threshold

    def compute(self, view: DataView) -> dict[str, float]:
        scores = {}

        for symbol in view.symbols:
            volumes = view.volume(symbol, self.lookback + 1)
            if len(volumes) < self.lookback + 1:
                continue

            avg_volume = np.mean(volumes[:-1])
            current_volume = volumes[-1]

            if avg_volume == 0:
                continue

            volume_ratio = current_volume / avg_volume

            if volume_ratio > self.spike_threshold:
                scores[symbol] = min((volume_ratio - 1) / self.spike_threshold, 1.0)
            else:
                scores[symbol] = 0.0

        return scores


class PriceDistanceMAFactor(Factor):
    def __init__(self, lookback: int = 50):
        self.lookback = lookback

    def compute(self, view: DataView) -> dict[str, float]:
        scores = {}

        for symbol in view.symbols:
            prices = view.close(symbol, self.lookback)
            if len(prices) < self.lookback:
                continue

            ma = np.mean(prices)
            current_price = prices[-1]

            distance = (current_price - ma) / ma

            scores[symbol] = distance

        return scores


class HighLowRangeFactor(Factor):
    def __init__(self, lookback: int = 20):
        self.lookback = lookback

    def compute(self, view: DataView) -> dict[str, float]:
        scores = {}

        for symbol in view.symbols:
            highs = view.high(symbol, self.lookback)
            lows = view.low(symbol, self.lookback)
            closes = view.close(symbol, self.lookback)

            if len(highs) < self.lookback:
                continue

            highest = np.max(highs)
            lowest = np.min(lows)
            current = closes[-1]

            if highest == lowest:
                continue

            range_position = (current - lowest) / (highest - lowest)

            scores[symbol] = (range_position - 0.5) * 2

        return scores


class DownsideVolatilityFactor(Factor):
    def __init__(self, lookback: int = 30):
        self.lookback = lookback

    def compute(self, view: DataView) -> dict[str, float]:
        scores = {}

        for symbol in view.symbols:
            prices = view.close(symbol, self.lookback)
            if len(prices) < self.lookback:
                continue

            returns = np.diff(prices) / prices[:-1]
            downside_returns = returns[returns < 0]

            if len(downside_returns) == 0:
                scores[symbol] = 1.0
            else:
                downside_vol = np.std(downside_returns)
                scores[symbol] = -downside_vol

        return scores


class MoneyFlowFactor(Factor):
    def __init__(self, lookback: int = 14):
        self.lookback = lookback

    def compute(self, view: DataView) -> dict[str, float]:
        scores = {}

        for symbol in view.symbols:
            highs = view.high(symbol, self.lookback)
            lows = view.low(symbol, self.lookback)
            closes = view.close(symbol, self.lookback)
            volumes = view.volume(symbol, self.lookback)

            if len(closes) < self.lookback:
                continue

            typical_prices = (highs + lows + closes) / 3
            money_flow = typical_prices * volumes

            positive_flow = 0.0
            negative_flow = 0.0

            for i in range(1, len(typical_prices)):
                if typical_prices[i] > typical_prices[i - 1]:
                    positive_flow += money_flow[i]
                elif typical_prices[i] < typical_prices[i - 1]:
                    negative_flow += money_flow[i]

            if negative_flow == 0:
                scores[symbol] = 1.0
            else:
                mfi = positive_flow / (positive_flow + negative_flow)
                scores[symbol] = (mfi - 0.5) * 2

        return scores


class VolumeTrendFactor(Factor):
    def __init__(self, lookback: int = 20):
        self.lookback = lookback

    def compute(self, view: DataView) -> dict[str, float]:
        scores = {}

        for symbol in view.symbols:
            volumes = view.volume(symbol, self.lookback)
            if len(volumes) < self.lookback:
                continue

            x = np.arange(len(volumes))
            if np.std(volumes) == 0:
                continue

            correlation = np.corrcoef(x, volumes)[0, 1]

            scores[symbol] = correlation

        return scores


class VolatilitySpikeFactor(Factor):
    def __init__(self, lookback: int = 30, spike_lookback: int = 5):
        self.lookback = lookback
        self.spike_lookback = spike_lookback

    def compute(self, view: DataView) -> dict[str, float]:
        scores = {}

        for symbol in view.symbols:
            prices = view.close(symbol, self.lookback)
            if len(prices) < self.lookback:
                continue

            returns = np.diff(prices) / prices[:-1]
            long_vol = np.std(returns)

            recent_returns = returns[-self.spike_lookback :]
            recent_vol = np.std(recent_returns)

            if long_vol == 0:
                continue

            vol_ratio = recent_vol / long_vol

            scores[symbol] = -vol_ratio

        return scores


class PriceAccelerationFactor(Factor):
    def __init__(self, lookback: int = 10):
        self.lookback = lookback

    def compute(self, view: DataView) -> dict[str, float]:
        scores = {}

        for symbol in view.symbols:
            prices = view.close(symbol, self.lookback)
            if len(prices) < self.lookback:
                continue

            returns = np.diff(prices) / prices[:-1]
            acceleration = np.diff(returns)

            if len(acceleration) < 2:
                continue

            avg_acceleration = np.mean(acceleration)

            scores[symbol] = avg_acceleration * 100

        return scores


class VolumeWeightedMomentumFactor(Factor):
    def __init__(self, lookback: int = 20):
        self.lookback = lookback

    def compute(self, view: DataView) -> dict[str, float]:
        scores = {}

        for symbol in view.symbols:
            prices = view.close(symbol, self.lookback)
            volumes = view.volume(symbol, self.lookback)

            if len(prices) < self.lookback:
                continue

            returns = np.diff(prices) / prices[:-1]
            volume_weights = volumes[1:] / np.sum(volumes[1:])

            weighted_return = np.sum(returns * volume_weights)

            scores[symbol] = weighted_return

        return scores


class ReverseRSIFactor(Factor):
    def __init__(self, lookback: int = 14, oversold: float = 30, overbought: float = 70):
        self.lookback = lookback
        self.oversold = oversold
        self.overbought = overbought

    def compute(self, view: DataView) -> dict[str, float]:
        scores = {}

        for symbol in view.symbols:
            prices = view.close(symbol, self.lookback + 1)
            if len(prices) < self.lookback + 1:
                continue

            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)

            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)

            if avg_loss == 0:
                rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))

            if rsi < self.oversold:
                scores[symbol] = (self.oversold - rsi) / self.oversold
            elif rsi > self.overbought:
                scores[symbol] = -(rsi - self.overbought) / (100 - self.overbought)
            else:
                scores[symbol] = 0.0

        return scores
