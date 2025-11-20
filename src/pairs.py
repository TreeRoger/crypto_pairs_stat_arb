import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from typing import List, Tuple


def _ols_alpha_beta(y: pd.Series, x: pd.Series):
    # OLS regression: y_t = a + b * x_t + e_t
    df = pd.concat([y, x], axis=1).dropna()
    if len(df) < 50:
        return np.nan, np.nan

    X = sm.add_constant(df.iloc[:, 1].values)
    model = sm.OLS(df.iloc[:, 0].values, X).fit()
    a, b = model.params[0], model.params[1]
    return float(a), float(b)


def _adf_pvalue(series: pd.Series) -> tuple[float, float]:
    # Augmented Dickey-Fuller test on a spread series.
    s = series.dropna()
    if len(s) < 50:
        return (np.nan, np.nan)

    try:
        res = adfuller(s)
        return float(res[0]), float(res[1])
    except Exception:
        return (np.nan, np.nan)


def _half_life(spread: pd.Series) -> float:
    """
    Estimate half-life of mean reversion using AR(1):
        spread_t = c + rho * spread_{t-1} + e_t
    Returns half-life in bars (hours in your CoinGecko setup).
    If invalid, returns +inf.
    """
    s = spread.dropna()
    if len(s) < 50:
        return np.inf

    s_lag = s.shift(1).dropna()
    s_curr = s.loc[s_lag.index]

    X = sm.add_constant(s_lag.values)
    try:
        rho = sm.OLS(s_curr.values, X).fit().params[1]
    except Exception:
        return np.inf

    if rho <= 0 or rho >= 1:
        return np.inf

    return -np.log(2) / np.log(rho)


def select_pairs(
    px: pd.DataFrame,
    *,
    max_pairs: int = 20,
    per_asset_limit: int = 2,
    pmax: float = 0.10,           # ADF p-value threshold
    min_corr: float = 0.20,       # minimum return correlation
    min_obs: int = 500,           # minimum data points
    min_half_life: float = 5.0,   # in bars (hours here)
    max_half_life: float = 400.0,
    min_spread_var_ratio: float = 0.01,
    max_spread_var_ratio: float = 25.0,
) -> List[Tuple[str, str]]:

    px = px.copy().replace(0, np.nan).ffill()
    cols = [c for c in px.columns if px[c].notna().sum() >= min_obs]
    px = px[cols]

    if len(px.columns) < 2:
        return []

    # Work in log prices
    logp = np.log(px)
    rets = logp.diff().dropna()

    candidates = []
    n = len(cols)

    for i in range(n):
        a = cols[i]
        for j in range(i + 1, n):
            b = cols[j]

            lp = logp[[a, b]].dropna()
            if len(lp) < min_obs:
                continue

            r = rets[[a, b]].dropna()
            if len(r) < min_obs:
                continue

            corr = r[a].corr(r[b])
            if pd.isna(corr) or corr < min_corr:
                continue

            # Try both regression directions: y~x and x~y
            stats = []
            for y_col, x_col in [(b, a), (a, b)]:
                a_hat, b_hat = _ols_alpha_beta(lp[y_col], lp[x_col])
                if np.isnan(a_hat) or np.isnan(b_hat):
                    continue

                spread = lp[y_col] - (a_hat + b_hat * lp[x_col])

                # ADF stationarity test
                adf_stat, pval = _adf_pvalue(spread)
                if pd.isna(pval) or pval > pmax:
                    continue

                # Half-life
                hl = _half_life(spread)
                if not (min_half_life <= hl <= max_half_life):
                    continue

                # Spread variance ratio
                var_ratio = spread.var() / lp[y_col].var()
                if not (min_spread_var_ratio <= var_ratio <= max_spread_var_ratio):
                    continue

                stats.append({
                    "pair": (y_col, x_col),
                    "adf_stat": adf_stat,
                    "pval": pval,
                    "half_life": hl,
                    "corr": corr,
                    "var_ratio": var_ratio,
                })

            if stats:
                # More negative ADF, shorter half-life is better
                best = sorted(stats, key=lambda d: (d["adf_stat"], d["half_life"]))[0]
                candidates.append(best)

    if not candidates:
        return []

    # Global ranking
    candidates.sort(key=lambda d: (d["adf_stat"], d["half_life"], -d["corr"]))

    # Enforce per-asset limits and max_pairs
    used = {c: 0 for c in px.columns}
    selected: List[Tuple[str, str]] = []

    for d in candidates:
        y, x = d["pair"]
        if used[y] >= per_asset_limit or used[x] >= per_asset_limit:
            continue

        selected.append((y, x))
        used[y] += 1
        used[x] += 1

        if len(selected) >= max_pairs:
            break

    return selected
