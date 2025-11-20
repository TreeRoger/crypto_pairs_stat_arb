import os
import pandas as pd
from datetime import datetime
from data.coingecko_client import CoinGeckoClient

BASE_URL = "https://api.coingecko.com/api/v3/pro"

def download_ohlc(
    client: CoinGeckoClient,
    coin_id: str,
    vs_currency: str,
    days: str,
    interval: str,
    out_dir="data/coingecko_ohlc"
):

    os.makedirs(out_dir, exist_ok=True)

    url = f"{BASE_URL}/coins/{coin_id}/ohlc"
    params = {
        "vs_currency": vs_currency,
        "days": days,
        "precision": "full"
    }

    print(f"Downloading OHLC for {coin_id} ({interval})...")

    raw = client.get(url, params=params)

    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    outfile = os.path.join(out_dir, f"{coin_id}_{interval}.csv")
    df.to_csv(outfile, index=False)
    print(f"Saved {outfile}")
    return df
