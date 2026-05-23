"""Volume features.

OBV slope, MFI, A/D line proxy, dollar volume rank, up/down volume ratio.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Volume-derived features. Requires Volume + Close (+ High/Low/Open if available)."""
    out = pd.DataFrame(index=df.index)

    vol = df['Volume'].astype(float)
    close = df['Close']

    # Volume z-score (vs 60d mean/std)
    vol_mean = vol.rolling(60, min_periods=20).mean()
    vol_std = vol.rolling(60, min_periods=20).std()
    out['vol_zscore_60d'] = (vol - vol_mean) / vol_std

    # 5d vs 20d ratio (accumulation signal)
    out['vol_5_vs_20'] = vol.rolling(5).mean() / vol.rolling(20).mean()

    # Dollar volume rank within rolling window
    dollar_vol = vol * close
    out['vol_dollar_zscore_60d'] = (
        (dollar_vol - dollar_vol.rolling(60).mean()) / dollar_vol.rolling(60).std()
    )

    # OBV (On-Balance Volume)
    direction = np.sign(close.diff()).fillna(0)
    obv = (direction * vol).cumsum()
    out['vol_obv_slope_20'] = obv.diff(20) / vol.rolling(20).mean()  # normalized

    # Up/down volume ratio (rolling 20)
    up_vol = vol.where(direction > 0, 0).rolling(20).sum()
    down_vol = vol.where(direction < 0, 0).rolling(20).sum()
    out['vol_up_down_ratio_20'] = up_vol / down_vol.replace(0, np.nan)

    # Money Flow Index (MFI)
    if all(c in df.columns for c in ['High', 'Low']):
        typical = (df['High'] + df['Low'] + close) / 3
        money_flow = typical * vol
        positive_mf = money_flow.where(typical.diff() > 0, 0)
        negative_mf = money_flow.where(typical.diff() < 0, 0)
        pmf_14 = positive_mf.rolling(14).sum()
        nmf_14 = negative_mf.rolling(14).sum()
        mf_ratio = pmf_14 / nmf_14.replace(0, np.nan)
        out['vol_mfi_14'] = 100 - (100 / (1 + mf_ratio))

    # VWAP deviation (rolling 20d)
    if all(c in df.columns for c in ['High', 'Low']):
        typical = (df['High'] + df['Low'] + close) / 3
        vwap = (typical * vol).rolling(20).sum() / vol.rolling(20).sum()
        out['vol_vwap_dist'] = (close / vwap) - 1.0

    return out
