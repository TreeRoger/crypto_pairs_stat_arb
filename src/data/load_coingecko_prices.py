import os
import pandas as pd

def load_coingecko_hourly(directory="data/coingecko_hourly"):
    dfs = []

    for fname in os.listdir(directory):
        if fname.endswith(".csv"):
            path = os.path.join(directory, fname)
            df = pd.read_csv(path)
            coin = fname.replace("_hourly.csv", "")

            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp")["close"].rename(coin)
            dfs.append(df)

    px = pd.concat(dfs, axis=1).sort_index()
    px = px.ffill()

    return px
