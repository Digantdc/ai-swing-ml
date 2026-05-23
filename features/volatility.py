"""Volatility features.

Realized volatility over multiple windows, vol-of-vol, term-structure proxy,
and forward realized vol target (for the vol forecasting model).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def compute_volatility_features(
    df: pd.DataFrame,
    windows: tuple[int, ...] = (5, 21, 60),
) -> pd.DataFrame:
    """Annualized realized vol + vol regime features."""
    out = pd.DataFrame(index=df.index)
    log_ret = np.log(df['Close']).diff()

    # Realized vol at multiple windows (annualized)
    for w in windows:
        rv = log_ret.rolling(w).std() * np.sqrt(TRADING_DAYS)
        out[f'vol_rv_{w}d'] = rv

    # Vol ratio (short vs long) — term-structure proxy
    if 5 in windows and 60 in windows:
        out['vol_rv_5_vs_60'] = out['vol_rv_5d'] / out['vol_rv_60d']

    # Vol of vol
    if 21 in windows:
        out['vol_volofvol_21d'] = out['vol_rv_21d'].rolling(21).std()

    # Vol z-score (current vs 1y average)
    if 21 in windows:
        rv_mean = out['vol_rv_21d'].rolling(252).mean()
        rv_std = out['vol_rv_21d'].rolling(252).std()
        out['vol_rv_zscore_252d'] = (out['vol_rv_21d'] - rv_mean) / rv_std

    # Parkinson volatility (high-low based, more efficient estimator)
    if all(c in df.columns for c in ['High', 'Low']):
        log_hl = np.log(df['High'] / df['Low'])
        park = np.sqrt((1 / (4 * np.log(2))) * (log_hl ** 2).rolling(21).mean() * TRADING_DAYS)
        out['vol_parkinson_21d'] = park

    return out


def compute_forward_realized_vol(
    df: pd.DataFrame,
    horizon: int = 21,
) -> pd.Series:
    """Forward realized volatility target (annualized) — used as RF regression target."""
    log_ret = np.log(df['Close']).diff()
    # Forward window: from t+1 to t+horizon
    fwd_vol = log_ret.shift(-horizon).rolling(horizon).std() * np.sqrt(TRADING_DAYS)
    # Note: this is forward-looking, only use for training targets
    return fwd_vol
