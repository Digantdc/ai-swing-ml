"""Regime detector — rule-based classifier on macro state.

Outputs a discrete label and a continuous score in [-1, +1]:
    +1.0  trending bull (full risk-on)
     0.5  cautious bull
     0.0  chop / neutral
    -0.5  defensive
    -1.0  risk-off (max defense)

Inputs: macro time series — VIX, SPY close, optional breadth proxy.

Why rule-based and not ML:
    - The mapping from VIX/trend to "what kind of market we're in" is well-known
      and has been stable across decades.
    - Labeled regime data doesn't exist; you'd have to label it yourself.
    - Rules are transparent and adjustable for stress testing.
    - This output is then consumed by ML models downstream (ranker reweighting,
      position sizer regime scaling).

For v2, consider:
    - Markov-switching model (statsmodels.tsa.regime_switching)
    - KMeans on (VIX, SPY trend, breadth) — unsupervised regime discovery
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RegimeRules:
    # VIX thresholds
    vix_calm: float = 15.0
    vix_elevated: float = 22.0
    vix_panic: float = 30.0

    # SPY trend window
    spy_trend_window: int = 200
    # Look at SPY's slope over this window
    spy_slope_window: int = 21

    # Return over recent window for momentum confirmation
    spy_recent_return_window: int = 21


class RegimeDetector:
    """Classify market regime from macro state."""

    def __init__(self, rules: RegimeRules | None = None):
        self.rules = rules or RegimeRules()

    # -------------------------------------------------------------- detect

    def detect(
        self,
        macro_panel: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Compute regime time series from macro panel.

        Args:
            macro_panel: dict with at least 'SPY' and '^VIX' DataFrames.
                Optional: '^TNX', 'SMH' for additional signals.

        Returns:
            DataFrame indexed by date with columns:
                ['regime', 'regime_score', 'vix', 'spy_above_200ma',
                 'spy_21d_ret', 'spy_slope_21d', 'risk_on_pillars',
                 'risk_off_pillars']
        """
        if 'SPY' not in macro_panel or macro_panel['SPY'].empty:
            return pd.DataFrame()

        spy = macro_panel['SPY']['Close'].copy()
        spy.index = pd.to_datetime(spy.index)

        # SPY trend signals
        sma_200 = spy.rolling(self.rules.spy_trend_window, min_periods=50).mean()
        spy_above_200ma = (spy > sma_200).astype(int)
        spy_21d_ret = np.log(spy).diff(self.rules.spy_recent_return_window)
        # Slope of SPY over recent window (units: % per day)
        spy_slope = np.log(spy).diff() \
            .rolling(self.rules.spy_slope_window).mean()

        # VIX series (aligned to SPY index)
        if '^VIX' in macro_panel and not macro_panel['^VIX'].empty:
            vix = macro_panel['^VIX']['Close'].reindex(spy.index).ffill()
        else:
            vix = pd.Series(20.0, index=spy.index)  # neutral default

        # Count "risk-on" and "risk-off" pillars
        risk_on = pd.Series(0, index=spy.index, dtype=int)
        risk_off = pd.Series(0, index=spy.index, dtype=int)

        risk_on += (vix < self.rules.vix_calm).astype(int)
        risk_on += (spy_above_200ma == 1).astype(int)
        risk_on += (spy_21d_ret > 0.02).astype(int)
        risk_on += (spy_slope > 0).astype(int)

        risk_off += (vix > self.rules.vix_panic).astype(int)
        risk_off += (spy_above_200ma == 0).astype(int)
        risk_off += (spy_21d_ret < -0.03).astype(int)
        risk_off += (spy_slope < 0).astype(int)

        # Map to score: +1 if all 4 risk-on pillars, -1 if all 4 risk-off
        score = (risk_on - risk_off) / 4.0
        score = score.clip(-1.0, 1.0)

        # Discrete label
        def label(s: float) -> str:
            if s >= 0.75:
                return 'trending_bull'
            if s >= 0.25:
                return 'cautious_bull'
            if s > -0.25:
                return 'chop'
            if s > -0.75:
                return 'defensive'
            return 'risk_off'

        regime = score.apply(label)

        out = pd.DataFrame({
            'regime': regime,
            'regime_score': score,
            'vix': vix,
            'spy_above_200ma': spy_above_200ma,
            'spy_21d_ret': spy_21d_ret,
            'spy_slope_21d': spy_slope,
            'risk_on_pillars': risk_on,
            'risk_off_pillars': risk_off,
        }, index=spy.index)
        out.index.name = 'date'
        return out

    def latest(self, macro_panel: dict[str, pd.DataFrame]) -> dict:
        """Convenience — return today's regime as a flat dict."""
        df = self.detect(macro_panel)
        if df.empty:
            return {'regime': 'unknown', 'regime_score': 0.0}
        row = df.iloc[-1]
        return {
            'date': df.index[-1],
            'regime': row['regime'],
            'regime_score': float(row['regime_score']),
            'vix': float(row['vix']) if pd.notna(row['vix']) else None,
            'spy_above_200ma': int(row['spy_above_200ma']),
            'spy_21d_ret': float(row['spy_21d_ret']) if pd.notna(row['spy_21d_ret']) else None,
            'risk_on_pillars': int(row['risk_on_pillars']),
            'risk_off_pillars': int(row['risk_off_pillars']),
        }

    # ----------------------------------------------------- consumer helpers

    @staticmethod
    def score_to_exposure_multiplier(score: float) -> float:
        """Map regime score to a gross-exposure scaling factor in [0.25, 1.0].

        At +1.0 (trending bull) → 1.0× (full size)
        At  0.0 (chop)         → 0.75×
        At -1.0 (risk-off)     → 0.25× (defensive — quarter size)
        """
        # Linear interpolation: m = 0.625 + 0.375 * score (clipped)
        m = 0.625 + 0.375 * score
        return float(max(0.25, min(1.0, m)))

    @staticmethod
    def score_to_strategy_bias(score: float) -> dict:
        """Map regime to preferred options strategy biases.

        Returns dict of strategy → bias multiplier (1.0 baseline).
        Used by StrategySelector v2 if you want regime-aware overrides.
        """
        if score >= 0.5:  # bull
            return {
                'long_call': 1.2,
                'bull_call_spread': 1.0,
                'bull_put_credit': 1.0,
                'iron_condor': 0.5,  # don't sell premium in trending market
            }
        if score <= -0.5:  # risk-off
            return {
                'long_call': 0.3,
                'bull_call_spread': 0.5,
                'bull_put_credit': 0.3,  # don't sell puts during selloffs
                'iron_condor': 0.7,
            }
        # Chop
        return {
            'long_call': 0.7,
            'bull_call_spread': 0.8,
            'bull_put_credit': 1.0,
            'iron_condor': 1.2,  # range-bound, sell premium
        }
