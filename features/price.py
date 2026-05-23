"""Price/momentum features.

Includes:
- Log returns over multiple windows
- Cumulative returns and z-scores
- Drawdown from rolling high
- Streak counters (consecutive up/down)
- Distance from key moving averages in vol-adjusted units
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_price_features(
    df: pd.DataFrame,
    return_windows: list[int] = (1, 5, 10, 20, 60, 120),
    zscore_window: int = 252,
) -> pd.DataFrame:
    """Compute price/momentum features.

    Args:
        df: OHLCV DataFrame (must include 'Close')
        return_windows: lookback windows (trading days)
        zscore_window: rolling z-score normalization window

    Returns:
        DataFrame with added feature columns prefixed 'px_'.
    """
    out = pd.DataFrame(index=df.index)
    close = df['Close']
    log_close = np.log(close.replace(0, np.nan))

    # Log returns over various windows
    for w in return_windows:
        out[f'px_ret_{w}d'] = log_close.diff(w)

    # Z-score of returns (normalize for cross-sectional comparison)
    for w in (5, 21, 63):
        ret = log_close.diff(w)
        mean = ret.rolling(zscore_window).mean()
        std = ret.rolling(zscore_window).std()
        out[f'px_ret_{w}d_z'] = (ret - mean) / std

    # Drawdown from rolling 252d high
    rolling_max = close.rolling(252, min_periods=20).max()
    out['px_drawdown_252d'] = (close / rolling_max) - 1.0

    # Drawdown from rolling 21d high (short-term)
    rolling_max_21 = close.rolling(21, min_periods=5).max()
    out['px_drawdown_21d'] = (close / rolling_max_21) - 1.0

    # Distance above 200-day MA (vol-adjusted)
    ma200 = close.rolling(200, min_periods=50).mean()
    std200 = close.rolling(200, min_periods=50).std()
    out['px_dist_ma200_vol'] = (close - ma200) / std200

    # Streak counters (consecutive up days, consecutive down days)
    direction = np.sign(close.diff())
    streak = pd.Series(0, index=close.index)
    cur = 0
    for i, d in enumerate(direction.values):
        if np.isnan(d):
            cur = 0
        elif d > 0:
            cur = cur + 1 if cur > 0 else 1
        elif d < 0:
            cur = cur - 1 if cur < 0 else -1
        else:
            cur = 0
        streak.iloc[i] = cur
    out['px_streak'] = streak

    # Gap (open vs previous close)
    if 'Open' in df.columns:
        prev_close = close.shift(1)
        out['px_gap'] = (df['Open'] - prev_close) / prev_close

    # Range expansion (today's range vs 20d avg)
    if 'High' in df.columns and 'Low' in df.columns:
        rng = df['High'] - df['Low']
        avg_rng = rng.rolling(20).mean()
        out['px_range_ratio'] = rng / avg_rng

    return out
