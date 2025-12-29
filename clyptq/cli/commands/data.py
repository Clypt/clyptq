"""Download market data. Use --all to avoid look-ahead bias."""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd

from clyptq.data.loaders.ccxt import CCXTLoader


class DataCLI:
    """Data download CLI."""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.data_dir = self.project_root / "data"

    def get_top_symbols(
        self, exchange_id: str = "binance", quote: str = "USDT", limit: int = 60
    ) -> List[str]:
        """Get top N by volume."""
        print(f"\nFetching top {limit} {quote} pairs from {exchange_id}...")

        loader = CCXTLoader(exchange_id)

        all_symbols = loader.get_available_symbols(quote=quote)
        print(f"Found {len(all_symbols)} {quote} pairs")

        # Batch it or Binance yells 413 at you
        batch_size = 100
        all_tickers = {}

        print("Fetching 24h volume data...")
        for i in range(0, len(all_symbols), batch_size):
            batch = all_symbols[i : i + batch_size]
            try:
                batch_tickers = loader.exchange.fetch_tickers(batch)
                all_tickers.update(batch_tickers)
                print(f"  Processed {min(i + batch_size, len(all_symbols))}/{len(all_symbols)} symbols")
            except Exception as e:
                print(f"  Warning: Failed to fetch batch {i}-{i+batch_size}: {e}")
                continue

        symbol_volumes = []
        for symbol, ticker in all_tickers.items():
            if ticker.get("quoteVolume"):
                symbol_volumes.append((symbol, ticker["quoteVolume"]))

        symbol_volumes.sort(key=lambda x: x[1], reverse=True)
        top_symbols = [s[0] for s in symbol_volumes[:limit]]

        loader.close()

        print(f"\nTop {limit} symbols by 24h volume:")
        for i, (symbol, volume) in enumerate(symbol_volumes[:limit], 1):
            print(f"  {i:2d}. {symbol:15s} ${volume:,.0f}")

        return top_symbols

    def download_data(
        self,
        exchange_id: str = "binance",
        timeframe: str = "1d",
        days: int = 90,
        limit: int = 60,
        symbols: Optional[List[str]] = None,
        download_all: bool = False,
        market_type: str = "spot",
    ) -> None:
        """Download OHLCV data. Use download_all=True to avoid bias."""
        data_path = self.data_dir / market_type / exchange_id / timeframe
        data_path.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"Universe Data Download")
        print(f"{'='*70}")
        print(f"Exchange:    {exchange_id}")
        print(f"Timeframe:   {timeframe}")
        print(f"History:     {days} days")
        print(f"Save path:   {data_path}")
        print(f"{'='*70}\n")

        if symbols is not None:
            print(f"Using provided symbols: {symbols}")
        elif download_all:
            loader = CCXTLoader(exchange_id)
            symbols = loader.get_available_symbols(quote="USDT")
            loader.close()
            print(f"\nDownloading ALL {len(symbols)} USDT pairs")
            print("This prevents look-ahead bias in backtesting!")
        else:
            symbols = self.get_top_symbols(exchange_id, limit=limit)

        loader = CCXTLoader(exchange_id)
        since = datetime.now() - timedelta(days=days)

        print(f"\nDownloading {len(symbols)} symbols...")
        print(f"Period: {since.strftime('%Y-%m-%d')} to {datetime.now().strftime('%Y-%m-%d')}")
        print("-" * 70)

        success_count = 0
        failed = []

        for i, symbol in enumerate(symbols, 1):
            try:
                df = loader.load_ohlcv(symbol, timeframe=timeframe, since=since)
                filename = symbol.replace("/", "_") + ".parquet"
                filepath = data_path / filename

                df.to_parquet(filepath)

                print(
                    f"[{i:2d}/{len(symbols)}] {symbol:15s} "
                    f"{len(df):4d} bars  "
                    f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}  "
                    f"✓"
                )

                success_count += 1

            except Exception as e:
                print(f"[{i:2d}/{len(symbols)}] {symbol:15s} FAILED: {str(e)}")
                failed.append(symbol)
                continue

        loader.close()

        print("-" * 70)
        print(f"\nDownload Summary:")
        print(f"  Success: {success_count}/{len(symbols)}")
        print(f"  Failed:  {len(failed)}/{len(symbols)}")

        if failed:
            print(f"\nFailed symbols: {', '.join(failed)}")

        print(f"\nData saved to: {data_path}")
        print(f"Total size: {self._get_dir_size(data_path):.2f} MB")

    def list_data(self, exchange_id: str = "binance", timeframe: str = "1d", market_type: str = "spot") -> None:
        """List what you downloaded."""
        data_path = self.data_dir / market_type / exchange_id / timeframe

        if not data_path.exists():
            print(f"No data found at {data_path}")
            return

        files = list(data_path.glob("*.parquet"))

        if not files:
            print(f"No data files found at {data_path}")
            return

        print(f"\nData files in {data_path}:")
        print(f"{'='*70}")

        total_size = 0
        for filepath in sorted(files):
            size = filepath.stat().st_size
            total_size += size

            try:
                df = pd.read_parquet(filepath)
                start = df.index[0].strftime("%Y-%m-%d")
                end = df.index[-1].strftime("%Y-%m-%d")
                bars = len(df)

                symbol = filepath.stem.replace("_", "/")
                print(
                    f"{symbol:15s}  {bars:4d} bars  "
                    f"{start} to {end}  "
                    f"{size/1024:.1f} KB"
                )

            except Exception as e:
                print(f"{filepath.name:20s}  ERROR: {e}")

        print(f"{'='*70}")
        print(f"Total: {len(files)} files, {total_size/1024/1024:.2f} MB")

    def _get_dir_size(self, path: Path) -> float:
        """Directory size in MB."""
        total = 0
        for filepath in path.rglob("*"):
            if filepath.is_file():
                total += filepath.stat().st_size
        return total / 1024 / 1024


def handle_data(args):
    """Handle data command."""
    cli = DataCLI()

    if args.action == "download":
        cli.download_data(
            exchange_id=args.exchange,
            timeframe=args.timeframe,
            days=args.days,
            limit=args.limit,
            symbols=args.symbols,
            download_all=args.download_all,
            market_type=args.market,
        )
    elif args.action == "list":
        cli.list_data(
            exchange_id=args.exchange,
            timeframe=args.timeframe,
            market_type=args.market,
        )
