"""Macro features (VIX, SPY, sector ETFs, rates).

These are broadcast onto each ticker's data (everyone sees the same VIX today).
Plus we compute *relative* features (stock vs benchmark, stock vs sector).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_macro_features(
    macro_panel: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Build a macro feature DataFrame indexed by date.

    Args:
        macro_panel: dict of {ticker: OHLCV} for ['SPY', '^VIX', '^TNX', 'SMH', etc.]

    Returns:
        DataFrame indexed by date with columns prefixed 'mc_'.
    """
    # Pick a common index from SPY
    if 'SPY' not in macro_panel or macro_panel['SPY'].empty:
        return pd.DataFrame()

    idx = macro_panel['SPY'].index
    out = pd.DataFrame(index=idx)

    # SPY features
    spy_close = macro_panel['SPY']['Close']
    out['mc_spy_ret_5d'] = np.log(spy_close).diff(5)
    out['mc_spy_ret_21d'] = np.log(spy_close).diff(21)
    spy_sma200 = spy_close.rolling(200).mean()
    out['mc_spy_above_200ma'] = (spy_close > spy_sma200).astype(int)
    spy_rv21 = np.log(spy_close).diff().rolling(21).std() * np.sqrt(252)
    out['mc_spy_rv_21d'] = spy_rv21

    # VIX
    if '^VIX' in macro_panel:
        vix = macro_panel['^VIX']['Close'].reindex(idx).ffill()
        out['mc_vix'] = vix
        out['mc_vix_pct_52w'] = vix.rank(pct=True).rolling(252).mean()
        out['mc_vix_zscore_60d'] = (vix - vix.rolling(60).mean()) / vix.rolling(60).std()

    # 10Y treasury yield
    if '^TNX' in macro_panel:
        tnx = macro_panel['^TNX']['Close'].reindex(idx).ffill()
        out['mc_10y_yield'] = tnx
        out['mc_10y_change_21d'] = tnx.diff(21)

    # Yield curve slope (10Y - 5Y)
    if '^TNX' in macro_panel and '^FVX' in macro_panel:
        tnx = macro_panel['^TNX']['Close'].reindex(idx).ffill()
        fvx = macro_panel['^FVX']['Close'].reindex(idx).ffill()
        out['mc_yield_curve_slope'] = tnx - fvx

    # DXY (dollar index)
    if 'DX-Y.NYB' in macro_panel:
        dxy = macro_panel['DX-Y.NYB']['Close'].reindex(idx).ffill()
        out['mc_dxy_ret_21d'] = np.log(dxy).diff(21)

    # Sector ETF returns
    for etf in ('SMH', 'SOXX', 'IGV', 'BOTZ'):
        if etf in macro_panel:
            etf_close = macro_panel[etf]['Close'].reindex(idx).ffill()
            out[f'mc_{etf.lower()}_ret_21d'] = np.log(etf_close).diff(21)
            out[f'mc_{etf.lower()}_rel_spy_21d'] = (
                np.log(etf_close).diff(21) - np.log(spy_close).diff(21)
            )

    return out


def compute_relative_strength(
    stock_close: pd.Series,
    benchmark_close: pd.Series,
    windows: tuple[int, ...] = (5, 21, 63),
) -> pd.DataFrame:
    """Stock returns relative to benchmark over various windows."""
    aligned = pd.concat([stock_close, benchmark_close], axis=1, join='inner')
    aligned.columns = ['stock', 'bench']
    out = pd.DataFrame(index=aligned.index)
    log_stock = np.log(aligned['stock'])
    log_bench = np.log(aligned['bench'])
    for w in windows:
        out[f'rs_vs_bench_{w}d'] = log_stock.diff(w) - log_bench.diff(w)
    return out
