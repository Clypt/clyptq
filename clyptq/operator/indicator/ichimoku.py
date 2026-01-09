"""Ichimoku Cloud indicator."""

from clyptq import operator


def ichimoku(
    high,
    low,
    close,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
    displacement: int = 26
):
    """Ichimoku Cloud (Ichimoku Kinko Hyo).

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        tenkan_period: Tenkan-sen (Conversion Line) period (default 9)
        kijun_period: Kijun-sen (Base Line) period (default 26)
        senkou_b_period: Senkou Span B period (default 52)
        displacement: Cloud displacement (default 26)

    Returns:
        Dict with 'tenkan', 'kijun', 'senkou_a', 'senkou_b', 'chikou'
    """
    # Tenkan-sen (Conversion Line)
    tenkan_high = operator.ts_max(high, tenkan_period)
    tenkan_low = operator.ts_min(low, tenkan_period)
    tenkan = operator.div(operator.add(tenkan_high, tenkan_low), 2)

    # Kijun-sen (Base Line)
    kijun_high = operator.ts_max(high, kijun_period)
    kijun_low = operator.ts_min(low, kijun_period)
    kijun = operator.div(operator.add(kijun_high, kijun_low), 2)

    # Senkou Span A (Leading Span A) - shifted forward
    senkou_a = operator.ts_delay(
        operator.div(operator.add(tenkan, kijun), 2),
        -displacement  # negative shift = forward
    )

    # Senkou Span B (Leading Span B)
    senkou_b_high = operator.ts_max(high, senkou_b_period)
    senkou_b_low = operator.ts_min(low, senkou_b_period)
    senkou_b = operator.ts_delay(
        operator.div(operator.add(senkou_b_high, senkou_b_low), 2),
        -displacement
    )

    # Chikou Span (Lagging Span) - shifted backward
    chikou = operator.ts_delay(close, -displacement)

    return {
        "tenkan": tenkan,
        "kijun": kijun,
        "senkou_a": senkou_a,
        "senkou_b": senkou_b,
        "chikou": chikou,
    }


def tenkan_sen(high, low, period: int = 9):
    """Tenkan-sen (Conversion Line) only.

    Args:
        high: High prices
        low: Low prices
        period: Lookback period

    Returns:
        Tenkan-sen values
    """
    return operator.div(
        operator.add(operator.ts_max(high, period), operator.ts_min(low, period)),
        2
    )


def kijun_sen(high, low, period: int = 26):
    """Kijun-sen (Base Line) only.

    Args:
        high: High prices
        low: Low prices
        period: Lookback period

    Returns:
        Kijun-sen values
    """
    return operator.div(
        operator.add(operator.ts_max(high, period), operator.ts_min(low, period)),
        2
    )


def senkou_span_a(
    high,
    low,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    displacement: int = 26
):
    """Senkou Span A (Leading Span A) only.

    Args:
        high: High prices
        low: Low prices
        tenkan_period: Tenkan-sen period
        kijun_period: Kijun-sen period
        displacement: Forward shift

    Returns:
        Senkou Span A values
    """
    tenkan = tenkan_sen(high, low, tenkan_period)
    kijun = kijun_sen(high, low, kijun_period)
    return operator.ts_delay(
        operator.div(operator.add(tenkan, kijun), 2),
        -displacement
    )


def senkou_span_b(high, low, period: int = 52, displacement: int = 26):
    """Senkou Span B (Leading Span B) only.

    Args:
        high: High prices
        low: Low prices
        period: Lookback period
        displacement: Forward shift

    Returns:
        Senkou Span B values
    """
    return operator.ts_delay(
        operator.div(
            operator.add(operator.ts_max(high, period), operator.ts_min(low, period)),
            2
        ),
        -displacement
    )


def chikou_span(close, displacement: int = 26):
    """Chikou Span (Lagging Span) only.

    Args:
        close: Close prices
        displacement: Backward shift

    Returns:
        Chikou Span values
    """
    return operator.ts_delay(close, -displacement)
