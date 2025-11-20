import argparse
import pandas as pd
import os

from data.load_coingecko_prices import load_coingecko_hourly
from data import compute_returns
from pairs import select_pairs
from signals import generate_signals
from portfolio import positions_from_signals
from backtest import portfolio_returns, apply_costs
from metrics import (
    compute_turnover,
    compute_sharpe_ratio,
    compute_drawdown,
    compute_information_ratio,
    holding_period_estimate,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing CoinGecko hourly OHLC CSVs")
    parser.add_argument("--formation_frac", type=float, default=0.6)
    parser.add_argument("--window", type=int, default=90)
    parser.add_argument("--entry_z", type=float, default=1.0)
    parser.add_argument("--exit_z", type=float, default=0.2)
    parser.add_argument("--cost_bps", type=float, default=20)
    parser.add_argument("--max_pairs", type=int, default=12)
    parser.add_argument("--per_asset_limit", type=int, default=2)
    parser.add_argument("--min_corr", type=float, default=0.3)
    args = parser.parse_args()

    # Load CoinGecko hourly data
    print(f"Loading hourly OHLC close data from {args.data_dir} ...")
    px_full = load_coingecko_hourly(args.data_dir)

    if px_full.shape[1] < 2:
        raise ValueError("Need at least two tickers for pairs trading.")

    print(f"Loaded prices: shape={px_full.shape}, columns={list(px_full.columns)}")

    # Formation vs Test Split
    split_idx = int(len(px_full) * args.formation_frac)
    px_form = px_full.iloc[:split_idx].copy()
    px_test = px_full.iloc[split_idx:].copy()

    # Select cointegrated pairs
    pairs = select_pairs(
        px_form,
        max_pairs=args.max_pairs,
        per_asset_limit=args.per_asset_limit,
        pmax=0.05,
        min_corr=args.min_corr,
        min_obs=200,   # hourly data → increase min obs
    )

    print(f"Selected {len(pairs)} pairs: {pairs}")

    if not pairs:
        print("No cointegrated pairs found. Exiting.")
        pd.DataFrame({"net_ret": []}).to_csv("results_hourly_returns.csv")
        return

    # Generate signals on full data
    signals_full = generate_signals(px_full, pairs, window=args.window)

    # Slice signals to test window
    signals = signals_full.loc[px_test.index.min(): px_test.index.max()]

    # Portfolio construction
    weights = positions_from_signals(
        signals,
        pairs,
        entry_z=args.entry_z,
        exit_z=args.exit_z,
        max_gross=1.0,
        cap_per_pair=0.15,
    )

    # Backtest
    rets_full = compute_returns(px_full, log_returns=False)
    rets_test = rets_full.reindex(px_test.index)

    gross = portfolio_returns(weights, rets_test)
    turnover = compute_turnover(weights)
    net = apply_costs(gross, turnover, cost_bps=args.cost_bps)

    # Metrics
    sharpe = compute_sharpe_ratio(net)
    dd_series = compute_drawdown(net)
    max_dd = float(dd_series.min())
    benchmark = px_full.columns[0]
    ir = compute_information_ratio(net, benchmark=rets_test[benchmark])
    hp = holding_period_estimate(weights)

    print(
        f"Sharpe (net): {sharpe:.2f} | "
        f"Avg holding period (hours): {hp:.1f} | "
        f"Max drawdown: {max_dd * 100:.2f}% | "
        f"IR vs {benchmark}: {ir:.2f}"
    )

    # Save Results
    pd.DataFrame({"net_ret": net}).to_csv("results_hourly_returns.csv")
    print("Saved hourly returns to results_hourly_returns.csv")


if __name__ == "__main__":
    main()
