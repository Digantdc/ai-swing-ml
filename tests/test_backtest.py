"""Smoke tests for backtest engine and strategy selector."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.costs import CostModel, OptionsCostModel
from backtest.metrics import (
    annualized_return, sharpe_ratio, max_drawdown,
    hit_rate, information_coefficient, compute_metrics,
)
from backtest.options import bsm_call, bsm_put
from backtest.walk_forward import WalkForwardBacktest
from models.strategy_selector import StrategySelector


def test_cost_model():
    cm = CostModel(cost_per_leg=0.0015, slippage=0.001, live_drag=0.0005)
    assert cm.total_per_leg() == 0.003
    assert cm.round_trip_cost() == 0.006
    # 5% gross return -> ~4.4% net
    assert abs(cm.apply_to_return(0.05) - 0.044) < 1e-6


def test_options_cost_tier():
    om = OptionsCostModel()
    # Tier 1 (best) should cost less than tier 3
    assert om.cost_for_tier(1, n_legs=2) < om.cost_for_tier(3, n_legs=2)


def test_metrics_on_synthetic_returns():
    np.random.seed(0)
    # Profitable strategy: positive drift + noise
    ret = pd.Series(np.random.normal(0.001, 0.02, 252))
    sharpe = sharpe_ratio(ret)
    assert sharpe > 0, "expected positive Sharpe for positive-drift series"
    ann = annualized_return(ret)
    assert ann > 0
    mdd, _ = max_drawdown(ret)
    assert mdd <= 0


def test_bsm_call_put_parity():
    # ATM, 30 days, 30% vol
    c = bsm_call(100, 100, 30/365, 0.30)
    p = bsm_put(100, 100, 30/365, 0.30)
    # Put-call parity: C - P = S - K * exp(-rT) ~ 0 at ATM
    assert abs((c - p) - (100 - 100 * np.exp(-0.045 * 30/365))) < 0.01


def test_walk_forward_folds():
    wf = WalkForwardBacktest(initial_train_months=12, validation_months=2, retrain_freq_months=1)
    dates = pd.date_range('2022-01-01', '2026-01-01', freq='B')
    folds = wf.generate_folds(dates)
    assert len(folds) > 0
    for f in folds:
        # Train < valid < test in time
        assert f.train_end <= f.valid_start
        assert f.valid_end <= f.test_start


def test_strategy_selector_strong_bull_low_iv():
    sel = StrategySelector()
    trade = sel.select(
        ticker='NVDA', spot=200.0,
        direction_score_pct=0.85,
        predicted_vol=0.30, current_rv_21d=0.30,
        iv_rank=20, liquidity_tier=1,
    )
    assert trade.strategy == 'long_call'


def test_strategy_selector_strong_bull_high_iv():
    sel = StrategySelector()
    trade = sel.select(
        ticker='NVDA', spot=200.0,
        direction_score_pct=0.85,
        predicted_vol=0.50, current_rv_21d=0.40,
        iv_rank=70, liquidity_tier=1,
    )
    assert trade.strategy == 'bull_call_spread'


def test_strategy_selector_neutral_high_iv():
    sel = StrategySelector()
    trade = sel.select(
        ticker='NVDA', spot=200.0,
        direction_score_pct=0.50,
        predicted_vol=0.50, current_rv_21d=0.40,
        iv_rank=80, liquidity_tier=1,
    )
    assert trade.strategy == 'iron_condor'


def test_strategy_selector_tier4_returns_wait():
    sel = StrategySelector()
    trade = sel.select(
        ticker='SERV', spot=10.0,
        direction_score_pct=0.85,
        predicted_vol=0.50, current_rv_21d=0.40,
        iv_rank=50, liquidity_tier=4,
    )
    assert trade.strategy == 'wait'


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
