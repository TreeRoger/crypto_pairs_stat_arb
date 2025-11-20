# src/backtest.py

import pandas as pd


def portfolio_returns(weights: pd.DataFrame, asset_returns: pd.DataFrame) -> pd.Series:
    # Compute portfolio returns given previous-period weights and asset returns.
    # Align indices and columns
    asset_returns = asset_returns.reindex_like(weights).fillna(0.0)
    w_lag = weights.shift(1).fillna(0.0)

    port_ret = (w_lag * asset_returns).sum(axis=1)
    return port_ret


def apply_costs(
    gross_returns: pd.Series,
    turnover: pd.Series,
    cost_bps: float = 20.0,
) -> pd.Series:
    # Apply transaction costs proportional to turnover.

    cost_per_unit = cost_bps / 10_000.0
    trading_costs = turnover * cost_per_unit
    net = gross_returns - trading_costs
    return net
