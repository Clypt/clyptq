"""Test streaming data sources."""

import asyncio
from datetime import datetime

import pytest

from clypt.data.streaming.ccxt_stream import CCXTStreamingSource


@pytest.mark.asyncio
async def test_streaming_source_lifecycle():
    """Test start/stop cycle."""
    stream = CCXTStreamingSource("binance", poll_interval=0.1)

    ticks = []

    def on_tick(timestamp, prices):
        ticks.append((timestamp, prices))

    # Start
    task = asyncio.create_task(stream.start(["BTC/USDT"], on_tick))
    await asyncio.sleep(0.3)
    assert stream.is_running()

    # Stop
    await stream.stop()
    assert not stream.is_running()


@pytest.mark.asyncio
async def test_streaming_receives_prices():
    """Test price updates (real exchange, quick test)."""
    stream = CCXTStreamingSource("binance", poll_interval=0.2)

    ticks = []

    def on_tick(timestamp, prices):
        ticks.append(prices)

    task = asyncio.create_task(stream.start(["BTC/USDT"], on_tick))
    await asyncio.sleep(1.0)  # Wait a bit longer

    await stream.stop()

    # Should have received at least a few ticks
    assert len(ticks) >= 2, f"Expected >=2 ticks, got {len(ticks)}"
    if ticks:
        assert "BTC/USDT" in ticks[-1], "BTC/USDT price should be present"


@pytest.mark.asyncio
async def test_concurrent_symbol_fetching():
    """Test concurrent price fetching."""
    stream = CCXTStreamingSource("binance", poll_interval=0.3)

    symbols = ["BTC/USDT", "ETH/USDT"]
    ticks = []

    def on_tick(timestamp, prices):
        ticks.append(prices)

    task = asyncio.create_task(stream.start(symbols, on_tick))
    await asyncio.sleep(1.0)

    await stream.stop()

    # Should get prices for multiple symbols
    assert len(ticks) >= 2
    if ticks:
        last_tick = ticks[-1]
        assert len(last_tick) >= 1  # At least 1 symbol
