# src/metrics.py

import numpy as np
import pandas as pd


def compute_turnover(weights: pd.DataFrame) -> pd.Series:
    # Turnover_t = 0.5 * sum_i |w_{t,i} - w_{t-1,i}|
    dw = weights.diff().abs()
    turnover = 0.5 * dw.sum(axis=1)
    turnover = turnover.fillna(0.0)
    return turnover


def compute_sharpe_ratio(returns: pd.Series, periods_per_year: float = 24 * 365) -> float:
    # Annualized Sharpe ratio for crypto hourly data by default.
    r = returns.dropna()
    if len(r) < 2:
        return 0.0

    mean = r.mean()
    std = r.std(ddof=1)
    if std == 0:
        return 0.0

    return float((mean / std) * np.sqrt(periods_per_year))


def compute_drawdown(returns: pd.Series) -> pd.Series:
    # Computes drawdown series from returns.
    r = returns.fillna(0.0)
    equity = (1.0 + r).cumprod()
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return dd


def compute_information_ratio(
    returns: pd.Series,
    benchmark: pd.Series,
    periods_per_year: float = 24 * 365,
) -> float:
    # Information ratio: annualized mean of (r - b) divided by std of (r - b).
    aligned = pd.concat([returns, benchmark], axis=1).dropna()
    if aligned.shape[0] < 2:
        return 0.0

    diff = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    mean = diff.mean()
    std = diff.std(ddof=1)
    if std == 0:
        return 0.0

    return float((mean / std) * np.sqrt(periods_per_year))


def holding_period_estimate(weights: pd.DataFrame) -> float:
    # Rough estimate of average holding period in bars (hours for hourly data).
    # We count how many bars positions are non-zero, divided by number of entries.
    # binary indicator of being in any position
    in_pos = (weights.abs().sum(axis=1) > 1e-6).astype(int)
    if in_pos.sum() == 0:
        return 0.0

    # entries where position goes from 0 -> 1
    entries = ((in_pos.shift(1) == 0) & (in_pos == 1)).sum()
    if entries == 0:
        return float(in_pos.sum())

    avg_hp = in_pos.sum() / entries
    return float(avg_hp)
