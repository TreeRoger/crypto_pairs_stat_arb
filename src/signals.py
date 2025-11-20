import numpy as np
import pandas as pd


def _rolling_beta_alpha(y: pd.Series, x: pd.Series, window: int):

    # Rolling hedge ratio using simple linear relation:
    # beta_t = Cov(y, x) / Var(x)
    # alpha_t = E[y] - beta_t * E[x]
    cov = y.rolling(window).cov(x)
    varx = x.rolling(window).var()

    beta = cov / varx.replace(0.0, np.nan)
    my = y.rolling(window).mean()
    mx = x.rolling(window).mean()
    alpha = my - beta * mx

    beta = beta.replace([np.inf, -np.inf], np.nan).ffill()
    alpha = alpha.replace([np.inf, -np.inf], np.nan).ffill()

    return beta, alpha


def generate_signals(px: pd.DataFrame, pairs, window: int = 90) -> pd.DataFrame:
    # Generate rolling spread, z-score, and hedge ratios for each pair.
    logpx = np.log(px.replace(0, np.nan)).ffill()
    out = {}

    for (y_name, x_name) in pairs:
        y = logpx[y_name]
        x = logpx[x_name]

        beta, alpha = _rolling_beta_alpha(y, x, window=window)
        spread = y - (alpha + beta * x)

        mean = spread.rolling(window).mean()
        std = spread.rolling(window).std(ddof=0)
        std = std.where(std > 1e-8)  # avoid unstable z

        z = (spread - mean) / std

        out[(y_name, x_name, "beta")] = beta
        out[(y_name, x_name, "alpha")] = alpha
        out[(y_name, x_name, "spread")] = spread
        out[(y_name, x_name, "z")] = z

    signals = pd.DataFrame(out)
    signals = signals.dropna(how="all")
    return signals
