"""
CoinGecko API fetcher for cryptocurrency event study analysis
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os
from typing import List, Dict, Optional, Union
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class CoinGeckoFetcher:
    """Fetches cryptocurrency data from CoinGecko API for event studies."""

    COIN_IDS = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "BNB": "binancecoin",
        "ADA": "cardano",
        "XRP": "ripple",
        "LTC": "litecoin",
    }

    def __init__(self, api_key: Optional[str] = None, rate_limit: float = 1.2):
        self.api_key = api_key or os.getenv("COINGECKO_API_KEY")
        self.base_url = "https://api.coingecko.com/api/v3"
        self.rate_limit = rate_limit
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"x-cg-demo-api-key": self.api_key})

    def _request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make API request with rate limiting and error handling."""
        url = f"{self.base_url}/{endpoint}"

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            time.sleep(self.rate_limit)
            return response.json()

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                time.sleep(60)  # Rate limit hit, wait and retry
                return self._request(endpoint, params)
            raise

    def fetch_event_window(
        self,
        coins: List[str],
        event_date: Union[str, datetime],
        days_before: int = 30,
        days_after: int = 30,
        save_csv: bool = True,
        output_dir: str = "data/events",
    ) -> pd.DataFrame:
        """
        Fetch data around an event date.

        Args:
            coins: List of coin symbols
            event_date: Event date (YYYY-MM-DD or datetime)
            days_before/after: Window size
            save_csv: Save to CSV
            output_dir: Output directory

        Returns:
            DataFrame with OHLC, volume, market_cap
        """
        if isinstance(event_date, str):
            event_date = datetime.strptime(event_date, "%Y-%m-%d")

        start_date = event_date - timedelta(days=days_before)
        end_date = event_date + timedelta(days=days_after)

        all_data = []

        for coin in coins:
            coin_id = self.COIN_IDS.get(coin.upper())
            if not coin_id:
                continue

            try:
                data = self._fetch_coin_data(coin_id, start_date, end_date)
                data["symbol"] = coin.upper()
                all_data.append(data)
            except Exception as e:
                print(f"Error fetching {coin}: {e}")
                continue

        if all_data:
            df = pd.concat(all_data, ignore_index=True)
            df.sort_values(["symbol", "timestamp"], inplace=True)

            if save_csv:
                self._save_data(df, event_date, output_dir)

            return df

        return pd.DataFrame()

    def _fetch_coin_data(self, coin_id: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Fetch market data for a single coin and derive daily OHLC via resampling."""
        endpoint = f"coins/{coin_id}/market_chart/range"
        params = {"vs_currency": "usd", "from": int(start.timestamp()), "to": int(end.timestamp())}

        data = self._request(endpoint, params)

        # Prices to time series
        prices_df = pd.DataFrame(data.get("prices", []), columns=["timestamp", "price"])
        if prices_df.empty:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "market_cap"])

        prices_df["timestamp"] = pd.to_datetime(prices_df["timestamp"], unit="ms", utc=True)
        prices_df.set_index("timestamp", inplace=True)

        # Derive daily OHLC from prices
        ohlc = prices_df["price"].resample("1D").ohlc()

        # Volumes
        vols_df = pd.DataFrame(data.get("total_volumes", []), columns=["timestamp", "volume"])
        if not vols_df.empty:
            vols_df["timestamp"] = pd.to_datetime(vols_df["timestamp"], unit="ms", utc=True)
            vols_df.set_index("timestamp", inplace=True)
            volume_daily = vols_df["volume"].resample("1D").sum()
        else:
            volume_daily = pd.Series(index=ohlc.index, dtype=float)

        # Market caps
        caps_df = pd.DataFrame(data.get("market_caps", []), columns=["timestamp", "market_cap"])
        if not caps_df.empty:
            caps_df["timestamp"] = pd.to_datetime(caps_df["timestamp"], unit="ms", utc=True)
            caps_df.set_index("timestamp", inplace=True)
            mcap_daily = caps_df["market_cap"].resample("1D").last()
        else:
            mcap_daily = pd.Series(index=ohlc.index, dtype=float)

        # Combine
        daily = ohlc.join(volume_daily, how="left").join(mcap_daily, how="left")
        daily = daily.rename(columns={"volume": "volume", "market_cap": "market_cap"})
        daily = daily.reset_index()
        daily.rename(columns={"timestamp": "timestamp"}, inplace=True)

        return daily[["timestamp", "open", "high", "low", "close", "volume", "market_cap"]]

    def batch_fetch(
        self,
        coins: List[str],
        event_dates: List[Union[str, datetime]],
        days_before: int = 30,
        days_after: int = 30,
        output_dir: str = "data/events",
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple events."""
        results = {}

        for event_date in event_dates:
            df = self.fetch_event_window(coins, event_date, days_before, days_after, True, output_dir)

            if isinstance(event_date, datetime):
                event_date = event_date.strftime("%Y-%m-%d")

            results[event_date] = df
            time.sleep(5)  # Be respectful between events

        return results

    def _save_data(self, df: pd.DataFrame, event_date: datetime, output_dir: str):
        """Save data to CSV."""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)

        filename = f"event_{event_date.strftime('%Y%m%d')}.csv"
        df.to_csv(path / filename, index=False)


def fetch_daily_ohlc_coingecko(crypto: str, start: str, end: str, api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch daily OHLC data from CoinGecko API.

    Args:
        crypto: Cryptocurrency symbol (e.g., 'btc', 'eth')
        start: Start date in YYYY-MM-DD format
        end: End date in YYYY-MM-DD format
        api_key: Optional API key

    Returns:
        DataFrame with daily OHLC data (open, high, low, close)
    """
    fetcher = CoinGeckoFetcher(api_key=api_key)

    # Get coin ID
    coin_id = fetcher.COIN_IDS.get(crypto.upper())
    if not coin_id:
        raise ValueError(f"Unknown cryptocurrency: {crypto}")

    # Parse dates
    start_date = datetime.strptime(start, "%Y-%m-%d")
    end_date = datetime.strptime(end, "%Y-%m-%d")

    # Fetch hourly data from CoinGecko
    endpoint = f"coins/{coin_id}/market_chart/range"
    params = {"vs_currency": "usd", "from": int(start_date.timestamp()), "to": int(end_date.timestamp())}

    data = fetcher._request(endpoint, params)

    # Parse prices (CoinGecko returns hourly data for this range)
    df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)

    # Ensure timezone awareness (UTC)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    # Resample to daily OHLC
    # This aggregates hourly data into proper daily OHLC candles
    ohlc = df["price"].resample("1D").ohlc()

    # Handle any missing days by forward-filling
    ohlc = ohlc.ffill()

    # Ensure column names are lowercase
    ohlc.columns = [c.lower() for c in ohlc.columns]

    return ohlc
