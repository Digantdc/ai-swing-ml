"""Fundamental features from yfinance .info snapshot.

These are slow-moving (refresh weekly). We attach them as static columns
broadcast to every date in the OHLCV series.

Categories:
- Profitability: op margin, FCF margin, net margin
- Growth: EPS growth, revenue growth
- Valuation: P/E, P/S, PEG, EV/EBITDA, P/B
- Analyst: recommendation, target price upside
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_fundamental_features(
    info: dict,
    market_cap: float | None = None,
    current_price: float | None = None,
) -> dict:
    """Compute a flat dict of fundamental features from a yfinance .info dict.

    Returns dict with keys prefixed 'fnd_'. Missing data → NaN.
    """
    out = {}

    def _get(k, default=np.nan):
        v = info.get(k)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    # Profitability
    out['fnd_op_margin'] = _get('operatingMargins')
    out['fnd_profit_margin'] = _get('profitMargins')
    out['fnd_gross_margin'] = _get('grossMargins')
    out['fnd_roa'] = _get('returnOnAssets')
    out['fnd_roe'] = _get('returnOnEquity')

    # FCF margin
    fcf = _get('freeCashflow')
    rev = _get('totalRevenue')
    if rev and rev > 0 and not np.isnan(fcf):
        out['fnd_fcf_margin'] = fcf / rev
    else:
        out['fnd_fcf_margin'] = np.nan
    out['fnd_fcf_positive'] = 1.0 if (not np.isnan(fcf) and fcf > 0) else 0.0

    # Growth
    out['fnd_eps_growth_q'] = _get('earningsQuarterlyGrowth')
    out['fnd_eps_growth_y'] = _get('earningsGrowth')
    out['fnd_rev_growth'] = _get('revenueGrowth')

    # Valuation
    out['fnd_pe_trailing'] = _get('trailingPE')
    out['fnd_pe_forward'] = _get('forwardPE')
    out['fnd_peg'] = _get('pegRatio')
    out['fnd_ps'] = _get('priceToSalesTrailing12Months')
    out['fnd_pb'] = _get('priceToBook')
    out['fnd_ev_rev'] = _get('enterpriseToRevenue')
    out['fnd_ev_ebitda'] = _get('enterpriseToEbitda')

    # PEG quality flag (1 if 0 < PEG < 1.5 -> reasonable, else 0)
    peg = out['fnd_peg']
    out['fnd_peg_reasonable'] = 1.0 if (not np.isnan(peg) and 0 < peg < 1.5) else 0.0

    # Analyst
    out['fnd_analyst_mean'] = _get('recommendationMean')  # 1=Strong Buy, 5=Sell
    out['fnd_n_analysts'] = _get('numberOfAnalystOpinions')
    target_mean = _get('targetMeanPrice')
    if current_price and current_price > 0 and not np.isnan(target_mean):
        out['fnd_target_upside'] = (target_mean / current_price) - 1.0
    else:
        out['fnd_target_upside'] = np.nan

    # Short interest
    out['fnd_short_pct_float'] = _get('shortPercentOfFloat')

    # Beta
    out['fnd_beta'] = _get('beta')

    # 52w position
    high52 = _get('fiftyTwoWeekHigh')
    low52 = _get('fiftyTwoWeekLow')
    if current_price and high52 and low52 and high52 > low52:
        out['fnd_pct_of_52w_range'] = (current_price - low52) / (high52 - low52)
    else:
        out['fnd_pct_of_52w_range'] = np.nan

    return out


def attach_fundamentals_to_panel(
    ohlcv_panel: dict[str, pd.DataFrame],
    fundamentals_per_ticker: dict[str, dict],
) -> dict[str, pd.DataFrame]:
    """Broadcast static fundamentals onto each ticker's daily OHLCV.

    Returns a new dict of DataFrames with fundamental columns added.
    """
    enriched = {}
    for ticker, df in ohlcv_panel.items():
        info = fundamentals_per_ticker.get(ticker, {})
        cur_price = df['Close'].iloc[-1] if not df.empty else None
        feats = compute_fundamental_features(info, current_price=cur_price)
        enriched_df = df.copy()
        for k, v in feats.items():
            enriched_df[k] = v
        enriched[ticker] = enriched_df
    return enriched
