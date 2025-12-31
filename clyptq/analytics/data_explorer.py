from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from clyptq.data.stores.store import DataStore


class DataExplorer:
    def __init__(self, store: DataStore):
        self.store = store

    def statistical_summary(self, symbols: Optional[List[str]] = None) -> pd.DataFrame:
        if symbols is None:
            symbols = self.store.symbols

        summaries = []
        for symbol in symbols:
            df = self.store.get_ohlcv(symbol)
            if df.empty:
                continue

            returns = df["close"].pct_change().dropna()

            summary = {
                "symbol": symbol,
                "count": len(df),
                "mean_price": df["close"].mean(),
                "std_price": df["close"].std(),
                "min_price": df["close"].min(),
                "max_price": df["close"].max(),
                "mean_volume": df["volume"].mean(),
                "std_volume": df["volume"].std(),
                "mean_return": returns.mean(),
                "std_return": returns.std(),
                "skew": returns.skew(),
                "kurtosis": returns.kurtosis(),
            }
            summaries.append(summary)

        return pd.DataFrame(summaries)

    def correlation_matrix(
        self, symbols: Optional[List[str]] = None, window: Optional[int] = None
    ) -> pd.DataFrame:
        if symbols is None:
            symbols = self.store.symbols

        returns_data = {}
        for symbol in symbols:
            df = self.store.get_ohlcv(symbol)
            if df.empty:
                continue

            returns = df["close"].pct_change().dropna()
            if window:
                returns = returns.iloc[-window:]

            returns_data[symbol] = returns

        if not returns_data:
            return pd.DataFrame()

        returns_df = pd.DataFrame(returns_data)
        return returns_df.corr()

    def price_statistics(
        self, symbol: str, start: Optional[datetime] = None, end: Optional[datetime] = None
    ) -> Dict:
        df = self.store.get_ohlcv(symbol)
        if df.empty:
            return {}

        if start:
            df = df[df.index >= start]
        if end:
            df = df[df.index <= end]

        returns = df["close"].pct_change().dropna()

        return {
            "symbol": symbol,
            "period_start": df.index[0].isoformat() if len(df) > 0 else None,
            "period_end": df.index[-1].isoformat() if len(df) > 0 else None,
            "total_bars": len(df),
            "price_mean": float(df["close"].mean()),
            "price_std": float(df["close"].std()),
            "price_min": float(df["close"].min()),
            "price_max": float(df["close"].max()),
            "volume_mean": float(df["volume"].mean()),
            "volume_std": float(df["volume"].std()),
            "return_mean": float(returns.mean()),
            "return_std": float(returns.std()),
            "return_skew": float(returns.skew()),
            "return_kurtosis": float(returns.kurtosis()),
            "sharpe_ratio": float(returns.mean() / returns.std() * np.sqrt(252))
            if returns.std() > 0
            else 0.0,
        }

    def volume_profile(self, symbol: str, bins: int = 20) -> pd.DataFrame:
        df = self.store.get_ohlcv(symbol)
        if df.empty:
            return pd.DataFrame()

        price_min = df["close"].min()
        price_max = df["close"].max()
        bin_edges = np.linspace(price_min, price_max, bins + 1)

        volume_profile = []
        for i in range(bins):
            mask = (df["close"] >= bin_edges[i]) & (df["close"] < bin_edges[i + 1])
            volume_in_bin = df.loc[mask, "volume"].sum()

            volume_profile.append(
                {
                    "price_low": bin_edges[i],
                    "price_high": bin_edges[i + 1],
                    "volume": volume_in_bin,
                }
            )

        return pd.DataFrame(volume_profile)

    def seasonality_analysis(self, symbol: str) -> Dict[str, pd.Series]:
        df = self.store.get_ohlcv(symbol)
        if df.empty:
            return {}

        returns = df["close"].pct_change().dropna()

        monthly_returns = returns.groupby(returns.index.month).mean()
        weekday_returns = returns.groupby(returns.index.dayofweek).mean()

        return {"monthly": monthly_returns, "weekday": weekday_returns}

    def rolling_correlation(
        self, symbol1: str, symbol2: str, window: int = 20
    ) -> pd.Series:
        df1 = self.store.get_ohlcv(symbol1)
        df2 = self.store.get_ohlcv(symbol2)

        if df1.empty or df2.empty:
            return pd.Series()

        returns1 = df1["close"].pct_change().dropna()
        returns2 = df2["close"].pct_change().dropna()

        combined = pd.DataFrame({"r1": returns1, "r2": returns2}).dropna()

        return combined["r1"].rolling(window).corr(combined["r2"])
