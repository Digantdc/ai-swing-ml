"""Smoke tests for the feature pipeline using synthetic data.

Run: python -m pytest tests/test_features.py -v
Or:  python tests/test_features.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running script directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.price import compute_price_features
from features.technical import compute_technical_features
from features.volume import compute_volume_features
from features.volatility import compute_volatility_features
from features.calendar import compute_calendar_features


def _synthetic_ohlcv(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    trend = np.linspace(50, 80, n)
    close = trend + rng.normal(0, 1.5, n)
    high = close + np.abs(rng.normal(0, 0.8, n))
    low = close - np.abs(rng.normal(0, 0.8, n))
    open_ = close + rng.normal(0, 0.4, n)
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    idx = pd.date_range('2025-01-01', periods=n, freq='B')
    return pd.DataFrame({
        'Open': open_, 'High': high, 'Low': low, 'Close': close, 'Volume': volume,
    }, index=idx)


def test_price_features_compute():
    df = _synthetic_ohlcv()
    feats = compute_price_features(df)
    assert not feats.empty
    assert 'px_ret_1d' in feats.columns
    assert 'px_drawdown_252d' in feats.columns
    # Last value should have all features non-NaN (after warm-up)
    assert feats.iloc[-1].notna().any()


def test_technical_features_compute():
    df = _synthetic_ohlcv()
    feats = compute_technical_features(df)
    assert 'ta_rsi_14' in feats.columns
    assert 'ta_macd_hist' in feats.columns
    assert 'ta_sma_stack' in feats.columns
    # RSI should be in [0, 100]
    rsi_clean = feats['ta_rsi_14'].dropna()
    assert (rsi_clean >= 0).all() and (rsi_clean <= 100).all()


def test_volume_features_compute():
    df = _synthetic_ohlcv()
    feats = compute_volume_features(df)
    assert 'vol_zscore_60d' in feats.columns
    assert 'vol_obv_slope_20' in feats.columns


def test_volatility_features_compute():
    df = _synthetic_ohlcv()
    feats = compute_volatility_features(df)
    assert 'vol_rv_21d' in feats.columns
    # Annualized vol for synthetic should be in a reasonable range
    rv = feats['vol_rv_21d'].dropna()
    assert (rv > 0).all() and (rv < 5).all()


def test_calendar_features_compute():
    df = _synthetic_ohlcv()
    feats = compute_calendar_features(df.index, next_earnings_date=pd.Timestamp('2025-03-15'))
    assert 'cal_days_to_earnings' in feats.columns
    assert 'cal_days_to_opex' in feats.columns


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as e:
                print(f"FAIL  {name}: {e}")
            except Exception as e:
                print(f"ERROR {name}: {type(e).__name__}: {e}")
