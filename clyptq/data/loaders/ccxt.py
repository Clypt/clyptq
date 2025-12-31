"""
CCXT data loader for fetching market data from exchanges.

Downloads OHLCV data from cryptocurrency exchanges via CCXT.
"""

from datetime import datetime, timedelta
from typing import List, Optional

import ccxt
import pandas as pd

from clyptq.data.stores.store import DataStore
from clyptq.infra.utils import get_logger


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
        self.exchange_id = exchange_id
        self.logger = get_logger(__name__, context={"exchange": exchange_id, "sandbox": sandbox})

        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class(
            {
                "enableRateLimit": True,
                "sandbox": sandbox,
            }
        )

        try:
            self.exchange.load_markets()
            self.logger.info("Exchange markets loaded", extra={"market_count": len(self.exchange.markets)})
        except ccxt.NetworkError as e:
            self.logger.error("Network error loading markets", extra={"error": str(e)})
            raise
        except ccxt.ExchangeError as e:
            self.logger.error("Exchange error loading markets", extra={"error": str(e)})
            raise

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

        try:
            while True:
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol=symbol, timeframe=timeframe, since=since_ms, limit=fetch_limit
                )

                if not ohlcv:
                    break

                all_ohlcv.extend(ohlcv)

                if len(ohlcv) < fetch_limit:
                    break

                since_ms = ohlcv[-1][0] + 1

        except ccxt.NetworkError as e:
            self.logger.error("Network error fetching OHLCV", extra={"symbol": symbol, "timeframe": timeframe, "error": str(e)})
            raise
        except ccxt.ExchangeError as e:
            self.logger.error("Exchange error fetching OHLCV", extra={"symbol": symbol, "timeframe": timeframe, "error": str(e)})
            raise

        if not all_ohlcv:
            self.logger.warning("No data returned for symbol", extra={"symbol": symbol, "timeframe": timeframe})
            raise ValueError(f"No data returned for {symbol}")

        df = pd.DataFrame(
            all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

        df.drop_duplicates(subset=["timestamp"], keep="first", inplace=True)

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

        self.logger.info("OHLCV data loaded", extra={"symbol": symbol, "timeframe": timeframe, "bars": len(df)})
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
                self.logger.info("Symbol loaded successfully", extra={"symbol": symbol, "bars": len(df)})

            except ccxt.NetworkError as e:
                self.logger.error("Network error loading symbol", extra={"symbol": symbol, "error": str(e)})
                continue
            except ccxt.ExchangeError as e:
                self.logger.warning("Exchange error loading symbol", extra={"symbol": symbol, "error": str(e)})
                continue
            except ValueError as e:
                self.logger.warning("No data for symbol", extra={"symbol": symbol, "error": str(e)})
                continue
            except Exception as e:
                self.logger.error("Unexpected error loading symbol", extra={"symbol": symbol, "error": str(e)})
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
