"""Kelly position sizer with regime scaling and caps.

Theory (one sentence):
    Kelly fraction f* = edge / variance.
    For correlated multi-asset portfolios with N positions, naive Kelly
    over-bets — fractional Kelly (0.25× or 0.5×) is the production norm.

This implementation:
    1. Takes expected edges (per-name 21d expected return)
       and predicted volatilities (per-name 21d annualized RV)
    2. Computes raw Kelly weights: edge / (vol²)
    3. Applies quarter-Kelly safety factor
    4. Scales gross exposure by regime score (defensive in risk-off)
    5. Caps per-position weight
    6. Normalizes to target gross exposure
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class KellySizerConfig:
    kelly_fraction: float = 0.25      # quarter-Kelly safety factor
    max_position_pct: float = 0.20    # max 20% in any single name
    base_gross_exposure: float = 1.0  # 100% gross at full risk-on
    min_gross_exposure: float = 0.25  # never below 25% even in risk-off
    edge_floor: float = 0.0           # ignore negative-edge positions
    vol_floor: float = 0.10           # min vol to avoid divide-by-near-zero
    vol_ceiling: float = 2.00         # cap vol input (some names spike to 200%)


class KellySizer:
    """Position-size top picks with edge/variance-aware weights."""

    def __init__(self, config: KellySizerConfig | None = None):
        self.cfg = config or KellySizerConfig()

    # --------------------------------------------------------------- size

    def size(
        self,
        edges: pd.Series,
        vols: pd.Series,
        regime_score: float = 0.0,
        gross_target: float | None = None,
    ) -> pd.Series:
        """Compute portfolio weights.

        Args:
            edges: index = ticker, values = expected 21d return (e.g. 0.04 for 4%).
                   For ranker-based pipelines, derive this by mapping score percentile
                   to empirical top-bucket return (see calibrate_edge_from_backtest).
            vols: index = ticker, values = predicted 21d annualized vol (e.g. 0.40).
            regime_score: -1 (risk-off) to +1 (trending bull). Scales gross exposure.
            gross_target: override for total gross exposure. If None, derived from regime.

        Returns:
            pd.Series indexed by ticker with portfolio weights (sum ≤ gross_target).
        """
        from .regime_detector import RegimeDetector  # avoid circular import at top

        if edges.empty:
            return pd.Series(dtype=float)
        aligned = pd.concat([edges, vols], axis=1, join='inner').dropna()
        aligned.columns = ['edge', 'vol']
        if aligned.empty:
            return pd.Series(dtype=float)

        # Sanitize
        aligned['edge'] = aligned['edge'].clip(lower=self.cfg.edge_floor)
        aligned['vol'] = aligned['vol'].clip(
            lower=self.cfg.vol_floor, upper=self.cfg.vol_ceiling
        )

        # Raw Kelly: f* = edge / variance
        # Note: variance = vol². At 40% vol → variance = 0.16 → 4% edge → f* = 25%
        kelly = aligned['edge'] / (aligned['vol'] ** 2)
        # Apply safety factor
        kelly = kelly * self.cfg.kelly_fraction
        # Per-position cap
        kelly = kelly.clip(upper=self.cfg.max_position_pct)
        # Drop zeros (no edge → no position)
        kelly = kelly[kelly > 0]
        if kelly.empty:
            return pd.Series(dtype=float)

        # Gross exposure: regime-scaled
        if gross_target is None:
            mult = RegimeDetector.score_to_exposure_multiplier(regime_score)
            gross_target = self.cfg.base_gross_exposure * mult
        gross_target = max(self.cfg.min_gross_exposure, gross_target)

        # If raw weights exceed target, scale down proportionally
        total = kelly.sum()
        if total > gross_target:
            kelly = kelly * (gross_target / total)

        return kelly.sort_values(ascending=False)

    # --------------------------------------------- edge calibration helpers

    @staticmethod
    def calibrate_edge_from_backtest(
        backtest_trades: pd.DataFrame,
        score_col: str = 'score',
        return_col: str = 'net_return',
        n_buckets: int = 10,
    ) -> dict[int, float]:
        """Build a mapping from score-bucket → empirical average return.

        Use the OUT-OF-SAMPLE backtest trades to estimate what return to
        expect for a given score percentile. This is more honest than
        guessing "top decile = 5% expected return."

        Returns:
            dict {bucket_index: avg_realized_return}.
            bucket_index 0 = lowest score decile, n_buckets-1 = highest.
        """
        if backtest_trades.empty or score_col not in backtest_trades.columns:
            return {}
        df = backtest_trades[[score_col, return_col]].dropna().copy()
        df['bucket'] = pd.qcut(df[score_col], n_buckets, labels=False, duplicates='drop')
        return df.groupby('bucket')[return_col].mean().to_dict()

    @staticmethod
    def score_pct_to_edge(
        score_pct: pd.Series,
        edge_calibration: dict[int, float] | None = None,
        default_top_edge: float = 0.04,
        default_bottom_edge: float = -0.02,
    ) -> pd.Series:
        """Map score percentile [0,1] to expected 21d return.

        If `edge_calibration` from a backtest is provided, use it.
        Otherwise linear interpolation between bottom and top defaults.
        """
        if edge_calibration:
            n = max(edge_calibration.keys()) + 1
            bucket = (score_pct * n).clip(0, n - 1).astype(int)
            return bucket.map(edge_calibration).astype(float).fillna(0.0)
        # Linear default
        return default_bottom_edge + score_pct * (default_top_edge - default_bottom_edge)

    # ---------------------------------------------------- portfolio metrics

    def expected_portfolio_stats(
        self,
        weights: pd.Series,
        edges: pd.Series,
        vols: pd.Series,
        assumed_pairwise_corr: float = 0.5,
    ) -> dict:
        """Estimate expected return, vol, Sharpe of the portfolio.

        Uses a simple correlation assumption (default 0.5 within AI sector)
        because the per-pair covariance matrix is high-dimensional and noisy.
        """
        aligned = pd.concat([weights, edges, vols], axis=1, join='inner').dropna()
        aligned.columns = ['w', 'edge', 'vol']
        if aligned.empty:
            return {'expected_return': 0.0, 'expected_vol': 0.0, 'expected_sharpe': 0.0}

        port_ret = (aligned['w'] * aligned['edge']).sum()
        # Approximate portfolio vol with constant pairwise correlation
        w = aligned['w'].values
        v = aligned['vol'].values
        diag = np.dot(w * v, w * v)  # sum of (w_i * v_i)^2
        cross = (w * v).sum() ** 2 - diag  # all cross-terms
        port_var = diag + assumed_pairwise_corr * cross
        port_vol = np.sqrt(max(port_var, 0))
        # 21d expected return → annualize: × (252/21)
        ann_ret = port_ret * (252 / 21)
        return {
            'expected_return_21d': float(port_ret),
            'expected_return_annualized': float(ann_ret),
            'expected_vol': float(port_vol),
            'expected_sharpe': float(ann_ret / port_vol) if port_vol > 0 else 0.0,
            'gross_exposure': float(aligned['w'].sum()),
            'n_positions': int(len(aligned)),
        }
