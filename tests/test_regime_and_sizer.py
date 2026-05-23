"""Tests for regime detector and Kelly position sizer."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.regime_detector import RegimeDetector, RegimeRules
from models.position_sizer import KellySizer, KellySizerConfig


# ---------------------------------------- synthetic macro data builders

def _build_macro_panel(n: int = 400, vix_level: float = 18, spy_drift: float = 0.0005, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2025-01-01', periods=n, freq='B')

    # SPY: drift + noise
    spy_ret = rng.normal(spy_drift, 0.01, n)
    spy_close = 400 * np.exp(np.cumsum(spy_ret))
    spy_df = pd.DataFrame({
        'Open': spy_close, 'High': spy_close * 1.005, 'Low': spy_close * 0.995,
        'Close': spy_close, 'Volume': rng.integers(50_000_000, 100_000_000, n),
    }, index=idx)

    # VIX: noisy around level
    vix_close = vix_level + rng.normal(0, 2, n)
    vix_close = np.clip(vix_close, 8, 80)
    vix_df = pd.DataFrame({
        'Open': vix_close, 'High': vix_close + 0.5, 'Low': vix_close - 0.5,
        'Close': vix_close, 'Volume': rng.integers(1_000, 10_000, n),
    }, index=idx)

    return {'SPY': spy_df, '^VIX': vix_df}


# ---------------------------------------- Regime detector tests

def test_regime_detector_bull_market():
    panel = _build_macro_panel(vix_level=14, spy_drift=0.0010)  # low VIX, strong drift
    rd = RegimeDetector()
    result = rd.detect(panel)
    assert not result.empty
    last = result.iloc[-1]
    # Should detect bull
    assert last['regime'] in ('trending_bull', 'cautious_bull')
    assert last['regime_score'] > 0


def test_regime_detector_risk_off():
    # High VIX + negative drift → risk-off
    panel = _build_macro_panel(vix_level=32, spy_drift=-0.0015)
    rd = RegimeDetector()
    result = rd.detect(panel)
    last = result.iloc[-1]
    assert last['regime'] in ('risk_off', 'defensive')
    assert last['regime_score'] < 0


def test_regime_detector_chop():
    # Mid VIX + slight positive drift → chop / cautious bull (depending on path)
    panel = _build_macro_panel(vix_level=20, spy_drift=0.0003, seed=7)
    rd = RegimeDetector()
    result = rd.detect(panel)
    last = result.iloc[-1]
    # Allow any non-extreme regime since the synthetic path is noisy
    assert last['regime'] in ('chop', 'cautious_bull', 'defensive', 'trending_bull')
    assert -1.0 <= last['regime_score'] <= 1.0
    # Just verify the field is populated and bounded


def test_regime_score_to_exposure_multiplier_bounds():
    # Extreme bear → minimum exposure
    assert RegimeDetector.score_to_exposure_multiplier(-1.0) == 0.25
    # Extreme bull → full exposure
    assert RegimeDetector.score_to_exposure_multiplier(+1.0) == 1.0
    # Mid → between
    mid = RegimeDetector.score_to_exposure_multiplier(0.0)
    assert 0.5 < mid < 0.85


def test_regime_strategy_bias_returns_dict():
    bias_bull = RegimeDetector.score_to_strategy_bias(0.8)
    assert 'long_call' in bias_bull
    assert bias_bull['long_call'] > 1.0  # favor long premium in bull
    bias_chop = RegimeDetector.score_to_strategy_bias(0.0)
    assert bias_chop['iron_condor'] > 1.0  # favor condors in chop


def test_regime_latest_returns_dict():
    panel = _build_macro_panel(vix_level=14, spy_drift=0.0008)
    rd = RegimeDetector()
    latest = rd.latest(panel)
    assert 'regime' in latest
    assert 'regime_score' in latest
    assert 'vix' in latest


# ---------------------------------------- Kelly sizer tests

def test_kelly_sizer_basic_allocation():
    sizer = KellySizer()
    # Two stocks: A has higher edge, B has higher vol
    edges = pd.Series({'A': 0.05, 'B': 0.03})
    vols = pd.Series({'A': 0.30, 'B': 0.50})
    weights = sizer.size(edges, vols, regime_score=1.0)
    assert weights['A'] > weights['B']  # A should get more weight
    assert weights.sum() <= 1.05  # within gross target


def test_kelly_sizer_caps_per_position():
    sizer = KellySizer(KellySizerConfig(max_position_pct=0.10))
    edges = pd.Series({'A': 0.50})  # absurd edge
    vols = pd.Series({'A': 0.20})
    weights = sizer.size(edges, vols, regime_score=1.0)
    assert weights['A'] <= 0.10 + 1e-9


def test_kelly_sizer_risk_off_shrinks_exposure():
    # Use strong edges so raw Kelly exceeds the bear-regime gross cap
    sizer = KellySizer()
    edges = pd.Series({'A': 0.10, 'B': 0.08, 'C': 0.06, 'D': 0.05, 'E': 0.04})
    vols = pd.Series({'A': 0.30, 'B': 0.30, 'C': 0.30, 'D': 0.30, 'E': 0.30})
    bull = sizer.size(edges, vols, regime_score=1.0).sum()
    bear = sizer.size(edges, vols, regime_score=-1.0).sum()
    assert bear < bull, f"bear ({bear:.3f}) should be < bull ({bull:.3f})"
    # Bear should be at the floor (~0.25) or below the bull total
    assert bear <= 0.30


def test_kelly_sizer_negative_edge_dropped():
    sizer = KellySizer()
    edges = pd.Series({'A': 0.05, 'B': -0.03})  # B has negative edge
    vols = pd.Series({'A': 0.30, 'B': 0.30})
    weights = sizer.size(edges, vols, regime_score=1.0)
    assert 'B' not in weights.index or weights.get('B', 0) == 0


def test_score_pct_to_edge_linear_default():
    scores = pd.Series([0.0, 0.5, 1.0])
    edges = KellySizer.score_pct_to_edge(scores)
    # Bottom score → bottom edge (-2%), top → top edge (+4%)
    assert abs(edges.iloc[0] - (-0.02)) < 1e-9
    assert abs(edges.iloc[-1] - 0.04) < 1e-9
    # Middle should be the average (within float tolerance)
    expected_mid = (edges.iloc[0] + edges.iloc[-1]) / 2
    assert abs(edges.iloc[1] - expected_mid) < 1e-9


def test_score_pct_to_edge_with_calibration():
    cal = {0: -0.04, 1: -0.01, 2: 0.01, 3: 0.03, 4: 0.06}
    scores = pd.Series([0.05, 0.45, 0.95])  # buckets 0, 2, 4
    edges = KellySizer.score_pct_to_edge(scores, edge_calibration=cal)
    assert abs(edges.iloc[0] - (-0.04)) < 1e-9
    assert abs(edges.iloc[-1] - 0.06) < 1e-9


def test_calibrate_edge_from_backtest():
    df = pd.DataFrame({
        'score': np.linspace(0, 1, 100),
        'net_return': np.linspace(-0.05, 0.08, 100),  # monotonic relationship
    })
    cal = KellySizer.calibrate_edge_from_backtest(df, n_buckets=5)
    assert len(cal) == 5
    # Top bucket return > bottom bucket
    assert cal[max(cal.keys())] > cal[min(cal.keys())]


def test_portfolio_stats_computes():
    sizer = KellySizer()
    weights = pd.Series({'A': 0.20, 'B': 0.15, 'C': 0.10})
    edges = pd.Series({'A': 0.04, 'B': 0.03, 'C': 0.02})
    vols = pd.Series({'A': 0.30, 'B': 0.30, 'C': 0.30})
    stats = sizer.expected_portfolio_stats(weights, edges, vols)
    assert stats['n_positions'] == 3
    assert stats['expected_return_annualized'] > 0
    assert stats['expected_sharpe'] > 0


# -----------------------------------------------------------------

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
