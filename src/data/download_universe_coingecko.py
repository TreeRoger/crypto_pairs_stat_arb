from data.coingecko_client import CoinGeckoClient
from data.download_ohlc_coingecko import download_ohlc

API_KEY = "API_KEY"

UNIVERSE = {
    "bitcoin": "BTC",
    "ethereum": "ETH",
    "solana": "SOL",
    "binancecoin": "BNB",
    "cardano": "ADA",
    "avalanche-2": "AVAX",
    "dogecoin": "DOGE",
    "chainlink": "LINK",
    "cosmos": "ATOM",
}

def main():
    client = CoinGeckoClient(API_KEY)

    for coin_id, symbol in UNIVERSE.items():
        download_ohlc(
            client,
            coin_id=coin_id,
            vs_currency="usd",
            days="max",
            interval="hourly",
            out_dir="data/coingecko_hourly"
        )

if __name__ == "__main__":
    main()
