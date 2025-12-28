"""
CLI tool for downloading and managing market data.

Usage:
    python -m clypt.cli.data download --exchange binance --days 90 --limit 60
    python -m clypt.cli.data list --exchange binance
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd

from clypt.data.loaders.ccxt import CCXTLoader


class DataCLI:
    """CLI for managing market data downloads."""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.data_dir = self.project_root / "data"

    def get_top_symbols(
        self, exchange_id: str = "binance", quote: str = "USDT", limit: int = 60
    ) -> List[str]:
        """
        Get top N symbols by 24h volume.

        Args:
            exchange_id: Exchange name
            quote: Quote currency
            limit: Number of symbols to return

        Returns:
            List of top symbols by volume
        """
        print(f"\nFetching top {limit} {quote} pairs from {exchange_id}...")

        loader = CCXTLoader(exchange_id)

        # Get all USDT pairs
        all_symbols = loader.get_available_symbols(quote=quote)
        print(f"Found {len(all_symbols)} {quote} pairs")

        # Fetch tickers in batches to avoid 413 error
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

        # Sort by quote volume (24h volume in quote currency)
        symbol_volumes = []
        for symbol, ticker in all_tickers.items():
            if ticker.get("quoteVolume"):
                symbol_volumes.append((symbol, ticker["quoteVolume"]))

        symbol_volumes.sort(key=lambda x: x[1], reverse=True)

        # Get top N
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
    ) -> None:
        """
        Download historical data for top symbols.

        Args:
            exchange_id: Exchange name
            timeframe: Data timeframe (e.g., '1d', '1h')
            days: Number of days of history
            limit: Number of symbols to download
            symbols: Specific symbols to download (overrides limit)
        """
        # Create data directory
        data_path = self.data_dir / exchange_id / timeframe
        data_path.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"Universe Data Download")
        print(f"{'='*70}")
        print(f"Exchange:    {exchange_id}")
        print(f"Timeframe:   {timeframe}")
        print(f"History:     {days} days")
        print(f"Save path:   {data_path}")
        print(f"{'='*70}\n")

        # Get symbols
        if symbols is None:
            symbols = self.get_top_symbols(exchange_id, limit=limit)
        else:
            print(f"Using provided symbols: {symbols}")

        # Download data
        loader = CCXTLoader(exchange_id)
        since = datetime.now() - timedelta(days=days)

        print(f"\nDownloading {len(symbols)} symbols...")
        print(f"Period: {since.strftime('%Y-%m-%d')} to {datetime.now().strftime('%Y-%m-%d')}")
        print("-" * 70)

        success_count = 0
        failed = []

        for i, symbol in enumerate(symbols, 1):
            try:
                # Download OHLCV
                df = loader.load_ohlcv(symbol, timeframe=timeframe, since=since)

                # Save to parquet
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

        # Summary
        print("-" * 70)
        print(f"\nDownload Summary:")
        print(f"  Success: {success_count}/{len(symbols)}")
        print(f"  Failed:  {len(failed)}/{len(symbols)}")

        if failed:
            print(f"\nFailed symbols: {', '.join(failed)}")

        print(f"\nData saved to: {data_path}")
        print(f"Total size: {self._get_dir_size(data_path):.2f} MB")

    def list_data(self, exchange_id: str = "binance", timeframe: str = "1d") -> None:
        """
        List downloaded data files.

        Args:
            exchange_id: Exchange name
            timeframe: Data timeframe
        """
        data_path = self.data_dir / exchange_id / timeframe

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

            # Load to check date range
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
        """Get directory size in MB."""
        total = 0
        for filepath in path.rglob("*"):
            if filepath.is_file():
                total += filepath.stat().st_size
        return total / 1024 / 1024


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Clypt Trading Engine - Data Management CLI"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Download command
    download_parser = subparsers.add_parser("download", help="Download market data")
    download_parser.add_argument(
        "--exchange", default="binance", help="Exchange name (default: binance)"
    )
    download_parser.add_argument(
        "--timeframe", default="1d", help="Timeframe (default: 1d)"
    )
    download_parser.add_argument(
        "--days", type=int, default=90, help="Days of history (default: 90)"
    )
    download_parser.add_argument(
        "--limit", type=int, default=60, help="Number of symbols (default: 60)"
    )
    download_parser.add_argument(
        "--symbols", nargs="+", help="Specific symbols to download"
    )

    # List command
    list_parser = subparsers.add_parser("list", help="List downloaded data")
    list_parser.add_argument(
        "--exchange", default="binance", help="Exchange name (default: binance)"
    )
    list_parser.add_argument(
        "--timeframe", default="1d", help="Timeframe (default: 1d)"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    cli = DataCLI()

    if args.command == "download":
        cli.download_data(
            exchange_id=args.exchange,
            timeframe=args.timeframe,
            days=args.days,
            limit=args.limit,
            symbols=args.symbols,
        )

    elif args.command == "list":
        cli.list_data(exchange_id=args.exchange, timeframe=args.timeframe)


if __name__ == "__main__":
    main()
