"""v2 smoke tests — universe, feature set, strategy selector, metrics fix.

Pure-Python tests (no xgboost needed). Run:
    python tests/test_v2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.training_universe import (
    get_training_tickers, get_trading_tickers, is_in_trading_universe,
    get_training_sector_map,
)
from features.feature_set import (
    get_v2_feature_list, get_v2_feature_list_with_regime, filter_to_v2,
)
from models.strategy_selector import StrategySelector
from backtest.metrics import (
    annualized_return, sharpe_ratio, compute_metrics,
)


# ---------------- Universe tests ----------------

def test_training_universe_larger_than_trading():
    train = get_training_tickers()
    trade = get_trading_tickers()
    assert len(train) > len(trade)
    assert len(train) >= 140, f"expected ~150 training names, got {len(train)}"
    assert len(trade) == 60 or len(trade) >= 55, f"expected ~60 trading names, got {len(trade)}"


def test_trading_subset_of_training():
    train = set(get_training_tickers())
    trade = set(get_trading_tickers())
    assert trade.issubset(train), "trading universe must be subset of training"


def test_is_in_trading_universe():
    assert is_in_trading_universe('NVDA')
    assert is_in_trading_universe('GFS')
    assert not is_in_trading_universe('JPM')   # JPM is training-only
    assert not is_in_trading_universe('XOM')   # energy diversifier


def test_sector_map_covers_all():
    train = get_training_tickers()
    smap = get_training_sector_map()
    for t in train:
        assert t in smap, f"{t} missing from sector map"


# ---------------- Feature set tests ----------------

def test_v2_feature_count():
    feats = get_v2_feature_list()
    assert 30 <= len(feats) <= 36, f"expected ~33 features, got {len(feats)}"


def test_v2_features_with_regime():
    feats = get_v2_feature_list_with_regime()
    assert 'regime_score' in feats
    assert 'regime_label_encoded' in feats


def test_v2_excludes_noise_technicals():
    feats = set(get_v2_feature_list())
    # These scored ~0.50 AUC (noise) and must be excluded
    for noise in ['ta_rsi_14', 'ta_macd', 'ta_macd_hist', 'ta_bb_pct',
                  'vol_vwap_dist', 'ta_stoch_k', 'ta_rsi_5']:
        assert noise not in feats, f"{noise} should be excluded (noise feature)"


def test_v2_includes_top_predictors():
    feats = set(get_v2_feature_list())
    # These were the strongest by IC
    for strong in ['mc_yield_curve_slope', 'mc_vix', 'fnd_roe', 'fnd_profit_margin']:
        assert strong in feats, f"{strong} should be included (top predictor)"


def test_filter_to_v2():
    cols = ['mc_vix', 'fnd_roe', 'ta_rsi_14', 'ticker', 'date', 'random_col']
    filtered = filter_to_v2(cols)
    assert 'mc_vix' in filtered
    assert 'fnd_roe' in filtered
    assert 'ta_rsi_14' not in filtered  # noise, excluded
    assert 'ticker' not in filtered     # metadata


# ---------------- Strategy selector v2 tests ----------------

def test_strategy_long_call_strong_bull_low_iv():
    sel = StrategySelector()
    t = sel.select('NVDA', 200.0, direction_prob=0.75, predicted_vol=0.30,
                   current_rv_21d=0.30, iv_rank=20, liquidity_tier=1)
    assert t.strategy == 'long_call'


def test_strategy_bull_call_spread_strong_bull_high_iv():
    sel = StrategySelector()
    t = sel.select('NVDA', 200.0, direction_prob=0.75, predicted_vol=0.55,
                   current_rv_21d=0.40, iv_rank=70, liquidity_tier=1)
    assert t.strategy == 'bull_call_spread'


def test_strategy_pmcc_moderate_bull_mid_iv_bull_regime():
    sel = StrategySelector()
    t = sel.select('NVDA', 200.0, direction_prob=0.65, predicted_vol=0.40,
                   current_rv_21d=0.38, iv_rank=45, liquidity_tier=1, regime_score=0.5)
    assert t.strategy == 'pmcc'


def test_strategy_bull_put_credit_moderate_bull_high_iv():
    sel = StrategySelector()
    t = sel.select('NVDA', 200.0, direction_prob=0.65, predicted_vol=0.55,
                   current_rv_21d=0.40, iv_rank=70, liquidity_tier=1, regime_score=0.0)
    assert t.strategy == 'bull_put_credit'


def test_strategy_earnings_iron_condor():
    sel = StrategySelector()
    t = sel.select('NVDA', 200.0, direction_prob=0.50, predicted_vol=0.55,
                   current_rv_21d=0.35, iv_rank=75, liquidity_tier=1, days_to_earnings=3)
    assert t.strategy == 'earnings_iron_condor'


def test_strategy_calendar_spread_neutral_low_iv():
    sel = StrategySelector()
    t = sel.select('NVDA', 200.0, direction_prob=0.50, predicted_vol=0.25,
                   current_rv_21d=0.30, iv_rank=20, liquidity_tier=1)
    assert t.strategy == 'calendar_spread'


def test_strategy_short_strangle_neutral_high_iv_lowvol_name():
    sel = StrategySelector()
    # neutral + high IV + low predicted vol + good liquidity → short strangle
    t = sel.select('AAPL', 200.0, direction_prob=0.50, predicted_vol=0.30,
                   current_rv_21d=0.22, iv_rank=70, liquidity_tier=1)
    assert t.strategy == 'short_strangle'


def test_strategy_iron_condor_neutral_high_iv_highvol_name():
    sel = StrategySelector()
    # neutral + high IV + HIGH predicted vol + good liquidity → iron condor (defined risk)
    t = sel.select('NVDA', 200.0, direction_prob=0.50, predicted_vol=0.55,
                   current_rv_21d=0.40, iv_rank=70, liquidity_tier=2)
    assert t.strategy == 'iron_condor'


def test_strategy_risk_off_regime_waits():
    sel = StrategySelector()
    t = sel.select('NVDA', 200.0, direction_prob=0.62, predicted_vol=0.40,
                   current_rv_21d=0.35, iv_rank=50, liquidity_tier=1, regime_score=-0.8)
    assert t.strategy == 'wait'


def test_strategy_tier4_waits():
    sel = StrategySelector()
    t = sel.select('SERV', 10.0, direction_prob=0.80, predicted_vol=0.50,
                   current_rv_21d=0.40, iv_rank=50, liquidity_tier=4)
    assert t.strategy == 'wait'


def test_short_strangle_flags_undefined_risk():
    sel = StrategySelector()
    t = sel.select('AAPL', 200.0, direction_prob=0.50, predicted_vol=0.30,
                   current_rv_21d=0.22, iv_rank=70, liquidity_tier=1)
    assert t.max_loss == float('inf')
    assert 'UNDEFINED RISK' in t.rationale


# ---------------- Metrics fix tests ----------------

def test_metrics_period_annualization():
    # 12 periods of 21 days each = ~1 year. 2% per period.
    returns = pd.Series([0.02] * 12)
    ann = annualized_return(returns, period_days=21)
    # (1.02)^12 - 1 ≈ 0.268 → ~27% annual
    assert 0.20 < ann < 0.35, f"expected ~27% annual, got {ann:.2%}"


def test_metrics_no_longer_explodes():
    # The v1 bug: 44 period-returns treated as 44 days → absurd annualization
    np.random.seed(0)
    returns = pd.Series(np.random.normal(0.01, 0.05, 44))
    sharpe = sharpe_ratio(returns, period_days=21)
    # Sharpe should be a sane number, not 6+
    assert -3 < sharpe < 4, f"sharpe {sharpe} is out of sane range — bug not fixed"


def test_compute_metrics_includes_period_info():
    returns = pd.Series([0.01, -0.02, 0.03, 0.01] * 5)
    m = compute_metrics(returns, period_days=21)
    assert m['period_days'] == 21
    assert m['n_periods'] == 20
    assert 'sharpe' in m and 'max_drawdown' in m


if __name__ == '__main__':
    passed = failed = 0
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
                passed += 1
            except AssertionError as e:
                print(f"FAIL  {name}: {e}")
                failed += 1
            except Exception as e:
                print(f"ERROR {name}: {type(e).__name__}: {e}")
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
