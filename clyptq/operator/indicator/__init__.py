"""Technical Indicators module.

This module provides a comprehensive set of technical indicators for
quantitative analysis. All indicators are implemented as stateless
pure functions that operate on pandas Series/DataFrame.

Usage:
    from clyptq.operator import indicator
    rsi_val = indicator.rsi(close, 14)

    # Or direct import
    from clyptq.operator.indicator import rsi, macd, bollinger_bands
"""

# Moving Averages
from clyptq.operator.indicator.sma import sma, ma
from clyptq.operator.indicator.ema import ema
from clyptq.operator.indicator.wma import wma
from clyptq.operator.indicator.dema import dema
from clyptq.operator.indicator.tema import tema

# Momentum Indicators
from clyptq.operator.indicator.rsi import rsi, stoch_rsi
from clyptq.operator.indicator.macd import (
    macd,
    macd_line,
    macd_signal,
    macd_histogram,
)
from clyptq.operator.indicator.stoch import stoch, fast_stoch, slow_stoch
from clyptq.operator.indicator.williams_r import williams_r
from clyptq.operator.indicator.cci import cci
from clyptq.operator.indicator.roc import roc, momentum

# Volatility Indicators
from clyptq.operator.indicator.bollinger import (
    bollinger_bands,
    bollinger_upper,
    bollinger_lower,
    bollinger_bandwidth,
    bollinger_percent_b,
)
from clyptq.operator.indicator.keltner import (
    keltner_channel,
    keltner_upper,
    keltner_lower,
)
from clyptq.operator.indicator.donchian import (
    donchian_channel,
    donchian_upper,
    donchian_lower,
    donchian_middle,
)
from clyptq.operator.indicator.atr import atr, natr, true_range

# Trend Indicators
from clyptq.operator.indicator.adx import adx, adx_value, plus_di, minus_di
from clyptq.operator.indicator.ichimoku import (
    ichimoku,
    tenkan_sen,
    kijun_sen,
    senkou_span_a,
    senkou_span_b,
    chikou_span,
)
from clyptq.operator.indicator.supertrend import (
    supertrend,
    supertrend_value,
    supertrend_direction,
)
from clyptq.operator.indicator.psar import psar, psar_value
from clyptq.operator.indicator.aroon import (
    aroon,
    aroon_up,
    aroon_down,
    aroon_oscillator,
)

# Volume Indicators
from clyptq.operator.indicator.obv import obv, obv_ema
from clyptq.operator.indicator.ad import ad, ad_oscillator
from clyptq.operator.indicator.cmf import cmf
from clyptq.operator.indicator.mfi import mfi
from clyptq.operator.indicator.vwap import vwap, vwap_rolling

# Price/Returns
from clyptq.operator.indicator.returns import (
    returns,
    log_returns,
    cumulative_returns,
    rolling_returns,
)

__all__ = [
    # Moving Averages
    "sma",
    "ma",
    "ema",
    "wma",
    "dema",
    "tema",
    # Momentum
    "rsi",
    "stoch_rsi",
    "macd",
    "macd_line",
    "macd_signal",
    "macd_histogram",
    "stoch",
    "fast_stoch",
    "slow_stoch",
    "williams_r",
    "cci",
    "roc",
    "momentum",
    # Volatility
    "bollinger_bands",
    "bollinger_upper",
    "bollinger_lower",
    "bollinger_bandwidth",
    "bollinger_percent_b",
    "keltner_channel",
    "keltner_upper",
    "keltner_lower",
    "donchian_channel",
    "donchian_upper",
    "donchian_lower",
    "donchian_middle",
    "atr",
    "natr",
    "true_range",
    # Trend
    "adx",
    "adx_value",
    "plus_di",
    "minus_di",
    "ichimoku",
    "tenkan_sen",
    "kijun_sen",
    "senkou_span_a",
    "senkou_span_b",
    "chikou_span",
    "supertrend",
    "supertrend_value",
    "supertrend_direction",
    "psar",
    "psar_value",
    "aroon",
    "aroon_up",
    "aroon_down",
    "aroon_oscillator",
    # Volume
    "obv",
    "obv_ema",
    "ad",
    "ad_oscillator",
    "cmf",
    "mfi",
    "vwap",
    "vwap_rolling",
    # Returns
    "returns",
    "log_returns",
    "cumulative_returns",
    "rolling_returns",
]
