"""
CCXT data loader for fetching market data from exchanges.

Downloads OHLCV data from cryptocurrency exchanges via CCXT.
"""

from datetime import datetime, timedelta
from typing import List, Optional

import ccxt
import pandas as pd

from clyptq.data.stores.store import DataStore


class CCXTLoader:
    """
    Load market data from exchanges via CCXT.

    Supports all CCXT-compatible exchanges.
    """

    def __init__(self, exchange_id: str = "binance", sandbox: bool = False):
        """
        Initialize CCXT loader.

        Args:
            exchange_id: Exchange identifier (e.g., 'binance', 'coinbase')
            sandbox: Use sandbox/testnet mode
        """
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class(
            {
                "enableRateLimit": True,
                "sandbox": sandbox,
            }
        )

        self.exchange_id = exchange_id
        self.exchange.load_markets()

    def load_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1d",
        since: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Load OHLCV data for a symbol.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe ('1m', '5m', '1h', '1d', etc.)
            since: Start date (if None, gets recent data)
            limit: Maximum number of candles per fetch (default: 1000)

        Returns:
            DataFrame with OHLCV data and DatetimeIndex

        Raises:
            ccxt.NetworkError: On network errors
            ccxt.ExchangeError: On exchange errors
        """
        all_ohlcv = []
        fetch_limit = limit or 1000

        since_ms = None
        if since:
            since_ms = int(since.timestamp() * 1000)

        # Fetch data in batches until we get all available data
        while True:
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=symbol, timeframe=timeframe, since=since_ms, limit=fetch_limit
            )

            if not ohlcv:
                break

            all_ohlcv.extend(ohlcv)

            # If we got less than the limit, we're done
            if len(ohlcv) < fetch_limit:
                break

            # Update since to the last timestamp + 1ms
            since_ms = ohlcv[-1][0] + 1

        if not all_ohlcv:
            raise ValueError(f"No data returned for {symbol}")

        # Convert to DataFrame
        df = pd.DataFrame(
            all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

        # Remove duplicates (can happen at batch boundaries)
        df.drop_duplicates(subset=["timestamp"], keep="first", inplace=True)

        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

        return df

    def load_multiple(
        self,
        symbols: List[str],
        timeframe: str = "1d",
        since: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> DataStore:
        """
        Load OHLCV data for multiple symbols into a DataStore.

        Args:
            symbols: List of trading pairs
            timeframe: Timeframe
            since: Start date
            limit: Maximum number of candles

        Returns:
            DataStore with all symbols loaded
        """
        store = DataStore()

        for symbol in symbols:
            try:
                df = self.load_ohlcv(symbol, timeframe, since, limit)
                store.add_ohlcv(symbol, df, frequency=timeframe, source=self.exchange_id)
                print(f"Loaded {symbol}: {len(df)} bars")

            except Exception as e:
                print(f"Failed to load {symbol}: {e}")
                continue

        return store

    def get_available_symbols(self, quote: str = "USDT") -> List[str]:
        """
        Get available trading pairs for a quote currency.

        Args:
            quote: Quote currency (e.g., 'USDT', 'USD')

        Returns:
            List of trading pairs
        """
        symbols = []

        for symbol in self.exchange.markets:
            if symbol.endswith(f"/{quote}"):
                symbols.append(symbol)

        return sorted(symbols)

    def close(self) -> None:
        """Close exchange connection."""
        if hasattr(self.exchange, "close"):
            self.exchange.close()


def load_crypto_data(
    symbols: List[str],
    exchange: str = "binance",
    timeframe: str = "1d",
    days: int = 365,
) -> DataStore:
    """
    Convenience function to load crypto data.

    Args:
        symbols: List of symbols (e.g., ['BTC/USDT', 'ETH/USDT'])
        exchange: Exchange name
        timeframe: Data timeframe
        days: Number of days to load

    Returns:
        DataStore with loaded data

    Example:
        >>> store = load_crypto_data(['BTC/USDT', 'ETH/USDT'], days=180)
    """
    loader = CCXTLoader(exchange)

    since = datetime.now() - timedelta(days=days)

    store = loader.load_multiple(symbols, timeframe=timeframe, since=since)

    loader.close()

    return store
