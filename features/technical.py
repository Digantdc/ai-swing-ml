"""Technical-indicator features (RSI, MACD, Bollinger, ADX, Stoch).

All computations are manual (no pandas-ta dep) for portability and reproducibility.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------- Core indicator helpers ----------------

def _rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _bollinger(series: pd.Series, length: int = 20, std: float = 2.0):
    middle = series.rolling(length).mean()
    rolling_std = series.rolling(length).std()
    upper = middle + std * rolling_std
    lower = middle - std * rolling_std
    pct = (series - lower) / (upper - lower)
    return upper, middle, lower, pct


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False, min_periods=length).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14):
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move
    tr_smooth = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1).ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/length, adjust=False, min_periods=length).mean() / tr_smooth
    minus_di = 100 * minus_dm.ewm(alpha=1/length, adjust=False, min_periods=length).mean() / tr_smooth
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    return adx, plus_di, minus_di


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3):
    lowest = low.rolling(k).min()
    highest = high.rolling(k).max()
    k_pct = 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    d_pct = k_pct.rolling(d).mean()
    return k_pct, d_pct


# ---------------- Feature compilation ----------------

def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical-indicator features to OHLCV df.

    Returns DataFrame of feature columns prefixed 'ta_'.
    """
    out = pd.DataFrame(index=df.index)
    close = df['Close']

    # SMAs / EMAs
    for n in (20, 50, 200):
        sma = close.rolling(n).mean()
        out[f'ta_sma_{n}_dist'] = (close / sma) - 1.0
    out['ta_ema_9_dist'] = (close / close.ewm(span=9, adjust=False).mean()) - 1.0
    out['ta_ema_21_dist'] = (close / close.ewm(span=21, adjust=False).mean()) - 1.0

    # SMA stack indicator (1 if full bull stack, -1 if full bear, 0 mixed)
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    bull_stack = (close > sma20) & (sma20 > sma50) & (sma50 > sma200)
    bear_stack = (close < sma20) & (sma20 < sma50) & (sma50 < sma200)
    out['ta_sma_stack'] = bull_stack.astype(int) - bear_stack.astype(int)

    # Golden cross signal (50 > 200 within last 60 days?)
    above = (sma50 > sma200).astype(int)
    crossed_up = (above.diff() == 1).astype(int)
    out['ta_golden_cross_60d'] = crossed_up.rolling(60).sum().clip(0, 1)

    # RSI (multiple lengths)
    out['ta_rsi_14'] = _rsi(close, 14)
    out['ta_rsi_5'] = _rsi(close, 5)

    # MACD
    macd, sig, hist = _macd(close)
    out['ta_macd'] = macd
    out['ta_macd_hist'] = hist
    out['ta_macd_above_signal'] = (macd > sig).astype(int)

    # Bollinger
    if all(c in df.columns for c in ['High', 'Low']):
        bbu, bbm, bbl, bbp = _bollinger(close)
        out['ta_bb_pct'] = bbp
        out['ta_bb_width'] = (bbu - bbl) / bbm  # squeeze/expansion

        # ATR (used by other modules too; expose normalized form here)
        atr = _atr(df['High'], df['Low'], close)
        out['ta_atr_pct'] = atr / close

        # ADX
        adx_val, plus_di, minus_di = _adx(df['High'], df['Low'], close)
        out['ta_adx'] = adx_val
        out['ta_di_diff'] = plus_di - minus_di

        # Stochastic
        k, d = _stochastic(df['High'], df['Low'], close)
        out['ta_stoch_k'] = k
        out['ta_stoch_d'] = d
        out['ta_stoch_diff'] = k - d

    return out
