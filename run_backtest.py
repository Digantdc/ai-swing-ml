#!/usr/bin/env python3
"""v2 walk-forward backtest — XGBoost binary classifier edition.

Key v2 changes vs v1:
  - Trains on ~150-name diverse universe (training_universe.py)
  - Predicts P(up_3pct_21d) with XGBoost binary classifier (not LambdaRank)
  - Merges regime_score / regime_label as broadcast features
  - Uses the curated 33-feature subset (feature_set.py)
  - Filters predictions to the 60-name trading universe before portfolio sim
  - Fixed metrics annualization (period_days param)

Usage:
    python run_backtest.py --config config.yaml --years 5
    python run_backtest.py --quick     # 2y data, faster smoke test
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data.fetcher import DataFetcher
from data.training_universe import (
    get_training_tickers, get_trading_tickers,
    get_training_sector_map, is_in_trading_universe,
)
from data.universe import get_liquidity_tier
from features.builder import FeatureBuilder
from features.feature_set import get_v2_feature_list, get_v2_feature_list_with_regime
from models.binary_classifier import BinaryDirectionClassifier
from models.strategy_selector import StrategySelector
from models.volatility_model import VolatilityModel
from models.regime_detector import RegimeDetector
from backtest.costs import CostModel, OptionsCostModel
from backtest.metrics import compute_metrics
from backtest.options import OptionsBacktester
from backtest.portfolio import PortfolioBacktester
from backtest.walk_forward import WalkForwardBacktest
from reports.daily import write_backtest_report

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('backtest_v2')


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def merge_regime(panel: pd.DataFrame, macro_panel: dict) -> pd.DataFrame:
    """Add regime_score + regime_label_encoded as broadcast features."""
    rd = RegimeDetector()
    regime_df = rd.detect(macro_panel)
    if regime_df.empty:
        panel['regime_score'] = 0.0
        panel['regime_label_encoded'] = 2  # 'chop' default
        return panel
    regime_df = regime_df.reset_index()
    regime_df['date'] = pd.to_datetime(regime_df['date'])
    # Encode regime label 0-4
    label_map = {'risk_off': 0, 'defensive': 1, 'chop': 2, 'cautious_bull': 3, 'trending_bull': 4}
    regime_df['regime_label_encoded'] = regime_df['regime'].map(label_map).fillna(2).astype(int)
    panel['date'] = pd.to_datetime(panel['date'])
    panel = panel.merge(
        regime_df[['date', 'regime_score', 'regime_label_encoded']],
        on='date', how='left',
    )
    panel['regime_score'] = panel['regime_score'].fillna(0.0)
    panel['regime_label_encoded'] = panel['regime_label_encoded'].fillna(2).astype(int)
    return panel


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config.yaml')
    p.add_argument('--years', type=int, default=None)
    p.add_argument('--quick', action='store_true', help='2y data smoke test')
    args = p.parse_args()

    cfg = load_config(args.config)
    years = args.years or cfg['backtest']['total_years']
    if args.quick:
        years = 2

    cache_dir = Path(cfg['output']['data_cache_dir'])
    out_dir = Path(cfg['output']['report_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    fetcher = DataFetcher(cache_dir)

    # --- Universe: train wide, trade narrow ---
    training_tickers = get_training_tickers()
    trading_tickers = set(get_trading_tickers())
    logger.info(f"Training universe: {len(training_tickers)} | Trading universe: {len(trading_tickers)}")

    benchmark = cfg['universe']['benchmark']
    sector_etfs = cfg['universe']['sector_etfs']
    macro_tickers = cfg['universe']['macro_tickers']

    end_date = datetime.now()
    start_date = end_date - timedelta(days=int(365 * (years + 1)))

    # --- Fetch ---
    logger.info(f"Fetching OHLCV for {len(training_tickers)} tickers, {years}y...")
    ohlcv_panel = fetcher.get_ohlcv_batch(training_tickers, start_date, end_date, progress=True)
    logger.info(f"Fetched {len(ohlcv_panel)} tickers successfully")

    macro_panel = {}
    for m in [benchmark] + sector_etfs + macro_tickers:
        df = fetcher.get_ohlcv(m, start_date, end_date)
        if not df.empty:
            macro_panel[m] = df

    logger.info("Fetching fundamentals...")
    fundamentals_panel = {}
    for i, t in enumerate(ohlcv_panel.keys(), 1):
        info = fetcher.get_fundamentals(t)
        if info:
            fundamentals_panel[t] = info
        if i % 25 == 0:
            logger.info(f"  fundamentals: {i}/{len(ohlcv_panel)}")
    earnings_dates = {t: fetcher.get_next_earnings_date(t) for t in ohlcv_panel.keys()}

    # --- Build features ---
    sector_map = get_training_sector_map()
    fb = FeatureBuilder(
        return_windows=tuple(cfg['features']['return_windows']),
        vol_windows=tuple(cfg['features']['vol_windows']),
        target_horizon=cfg['target']['horizon_days'],
        target_top_pct=cfg['target']['top_pct'],
        sector_map=sector_map,
    )
    logger.info("Building feature panel...")
    panel = fb.build_panel(ohlcv_panel, fundamentals_panel, macro_panel, earnings_dates, benchmark)
    panel = fb.add_targets(panel)
    panel = merge_regime(panel, macro_panel)
    logger.info(f"Panel shape: {panel.shape}")

    # --- Feature subset for classifier ---
    feature_subset = get_v2_feature_list_with_regime()
    available = [f for f in feature_subset if f in panel.columns]
    logger.info(f"Using {len(available)}/{len(feature_subset)} v2 features (rest missing from panel)")

    # --- Walk-forward ---
    wf = WalkForwardBacktest(
        initial_train_months=cfg['backtest']['initial_train_months'],
        validation_months=cfg['backtest']['validation_months'],
        retrain_freq_months=cfg['backtest']['retrain_freq_months'],
    )
    panel['date'] = pd.to_datetime(panel['date'])
    folds = wf.generate_folds(panel['date'].unique())
    logger.info(f"Generated {len(folds)} walk-forward folds")
    if not folds:
        logger.error("No folds — check date range vs initial_train_months")
        sys.exit(1)

    all_predictions = []
    all_vol_predictions = []
    feature_importance_acc = None
    target_threshold = cfg['target'].get('min_up_return', 0.03)

    for i, fold in enumerate(folds):
        logger.info(f"Fold {i+1}/{len(folds)}: test [{fold.test_start.date()}, {fold.test_end.date()}]")
        train, valid, test = wf.split_panel(panel, fold)
        if train.empty or test.empty:
            continue

        clf = BinaryDirectionClassifier(
            params=cfg['model'].get('xgboost', {}),
            num_boost_round=cfg['model'].get('xgboost', {}).get('num_boost_round', 500),
            early_stopping_rounds=cfg['model'].get('xgboost', {}).get('early_stopping_rounds', 50),
            random_state=cfg['model'].get('random_state', 42),
            calibrate=True,
            feature_subset=available,
        )
        try:
            clf.fit(train, valid if not valid.empty else None, target_threshold=target_threshold)
        except Exception as e:
            logger.warning(f"Fold {i+1} classifier fit failed: {e}")
            continue

        probs = clf.predict_proba(test)
        # Filter to trading universe only
        test_pred = test[['date', 'ticker']].copy()
        test_pred['score'] = probs.values  # P(up) used as score for ranking top-N
        test_pred = test_pred[test_pred['ticker'].isin(trading_tickers)]
        all_predictions.append(test_pred)

        # Vol model
        try:
            vm = VolatilityModel(**cfg['volatility_model']['rf'])
            vm.fit(train)
            vp = test[['date', 'ticker']].copy()
            vp['predicted_vol'] = vm.predict(test).values
            vp = vp[vp['ticker'].isin(trading_tickers)]
            all_vol_predictions.append(vp)
        except Exception as e:
            logger.warning(f"Fold {i+1} vol model failed: {e}")

        feature_importance_acc = clf.feature_importance()

    if not all_predictions:
        logger.error("No predictions produced — abort.")
        sys.exit(1)

    predictions = pd.concat(all_predictions, ignore_index=True)
    vol_forecasts = pd.concat(all_vol_predictions, ignore_index=True) if all_vol_predictions else None

    # --- Portfolio backtest (trading universe only) ---
    logger.info("Running portfolio backtest...")
    costs = CostModel(
        cost_per_leg=cfg['backtest']['cost_per_leg'],
        slippage=cfg['backtest']['slippage'],
        live_drag=cfg['realism']['live_drag'],
    )
    pbt = PortfolioBacktester(
        top_n=cfg['backtest']['top_n_picks'],
        rebalance_days=cfg['backtest']['rebalance_days'],
        cost_model=costs,
        weighting=cfg['backtest']['weighting'],
        min_score_pct=cfg['backtest']['min_signal_pct'],
    )
    # Only pass trading-universe prices
    trading_ohlcv = {t: df for t, df in ohlcv_panel.items() if t in trading_tickers}
    equity_curve, trades = pbt.run(predictions, trading_ohlcv)
    logger.info(f"Portfolio: {len(trades)} trades")

    # --- Options overlay ---
    options_pnl = None
    if cfg['options']['enabled'] and vol_forecasts is not None:
        logger.info("Running options overlay...")
        selector = StrategySelector(
            expiry_dte_short=cfg['options'].get('expiry_dte', 30),
            iv_to_rv_ratio=cfg['options']['iv_to_rv_ratio'],
            risk_per_trade=cfg['options']['risk_per_trade'],
        )
        obt = OptionsBacktester(
            strategy_selector=selector,
            cost_model=OptionsCostModel(),
            iv_to_rv_ratio=cfg['options']['iv_to_rv_ratio'],
            early_exit_pct=cfg['options']['early_exit_pct'],
        )
        liq_map = {t: get_liquidity_tier(t) for t in trading_ohlcv.keys()}
        try:
            options_pnl = obt.run(predictions, trading_ohlcv, vol_forecasts, liq_map)
            logger.info(f"Options overlay: {len(options_pnl)} trades")
        except Exception as e:
            logger.warning(f"Options overlay failed (equity metrics still computed): {e}")
            options_pnl = None

    # --- Metrics (with period_days fix) ---
    daily_returns = equity_curve['daily_return'] if 'daily_return' in equity_curve else pd.Series(dtype=float)
    pred_w_realized = predictions.merge(
        panel[['date', 'ticker', 'target_fwd_return']], on=['date', 'ticker'], how='left',
    )
    metrics = compute_metrics(
        daily_returns=daily_returns,
        trade_returns=trades['net_return'] if not trades.empty else None,
        predictions=pred_w_realized['score'],
        realized=pred_w_realized['target_fwd_return'],
        rf_annual=cfg['backtest']['cash_return_annual'],
        period_days=cfg['backtest']['rebalance_days'],
    )
    logger.info("=== METRICS (v2, corrected) ===")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")

    # --- Report ---
    report_path = write_backtest_report(
        metrics=metrics,
        feature_importance=feature_importance_acc if feature_importance_acc is not None else pd.DataFrame(),
        equity_curve=equity_curve,
        trades=trades,
        options_pnl=options_pnl,
        out_path=out_dir,
    )
    logger.info(f"Backtest report: {report_path}")


if __name__ == '__main__':
    main()
