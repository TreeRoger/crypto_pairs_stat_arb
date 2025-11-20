# src/portfolio.py

import pandas as pd
import numpy as np


def positions_from_signals(
    signals: pd.DataFrame,
    pairs,
    entry_z: float = 1.0,
    exit_z: float = 0.2,
    max_gross: float = 1.0,
    cap_per_pair: float = 0.15,
) -> pd.DataFrame:
    # Build a dollar-neutral portfolio from pair z-scores.
    # Prepare empty weights DataFrame with all asset columns
    assets = sorted(list({a for pair in pairs for a in pair}))
    weights = pd.DataFrame(0.0, index=signals.index, columns=assets)

    for (y_name, x_name) in pairs:
        z = signals[(y_name, x_name, "z")].copy()

        # +1 = long spread (long y, short x)
        # -1 = short spread (short y, long x)
        signal_state = pd.Series(0.0, index=z.index)

        # Enter long spread
        signal_state[z <= -entry_z] = 1.0
        # Enter short spread
        signal_state[z >= entry_z] = -1.0

        # Exit when |z| < exit_z => go flat
        signal_state[(z.abs() < exit_z)] = 0.0

        # Forward-fill positions (simple state machine)
        signal_state = signal_state.replace(to_replace=0.0, method="ffill").fillna(0.0)

        # Cap per pair
        pair_gross = cap_per_pair
        w_y = signal_state * pair_gross
        w_x = -signal_state * pair_gross  # dollar neutral

        weights[y_name] += w_y
        weights[x_name] += w_x

    # Enforce max gross exposure
    gross = weights.abs().sum(axis=1)
    excess = gross > max_gross
    if excess.any():
        scale = max_gross / gross
        scale[~excess] = 1.0
        weights = weights.mul(scale, axis=0)

    return weights
