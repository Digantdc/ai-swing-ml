#!/usr/bin/env python3
"""v2 daily inference — XGBoost classifier, filtered to trading universe.

Usage:
    python predict.py --config config.yaml --top-n 5

Outputs:
    output/daily_picks_YYYY-MM-DD.md
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data.fetcher import DataFetcher
from data.training_universe import get_training_tickers, get_training_sector_map, get_trading_tickers
from data.universe import get_liquidity_tier
from features.builder import FeatureBuilder
from features.feature_set import get_v2_feature_list_with_regime
from models.binary_classifier import BinaryDirectionClassifier
from models.strategy_selector import StrategySelector
from models.volatility_model import VolatilityModel
from models.regime_detector import RegimeDetector
from models.position_sizer import KellySizer
from reports.daily import write_daily_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger('predict_v2')


def merge_regime(panel, macro_panel):
    rd = RegimeDetector()
    regime_df = rd.detect(macro_panel)
    if regime_df.empty:
        panel['regime_score'] = 0.0
        panel['regime_label_encoded'] = 2
        return panel, {'regime': 'unknown', 'regime_score': 0.0}
    regime_df = regime_df.reset_index()
    regime_df['date'] = pd.to_datetime(regime_df['date'])
    label_map = {'risk_off': 0, 'defensive': 1, 'chop': 2, 'cautious_bull': 3, 'trending_bull': 4}
    regime_df['regime_label_encoded'] = regime_df['regime'].map(label_map).fillna(2).astype(int)
    panel['date'] = pd.to_datetime(panel['date'])
    panel = panel.merge(regime_df[['date', 'regime_score', 'regime_label_encoded']], on='date', how='left')
    panel['regime_score'] = panel['regime_score'].fillna(0.0)
    panel['regime_label_encoded'] = panel['regime_label_encoded'].fillna(2).astype(int)
    latest_regime = rd.latest(macro_panel)
    return panel, latest_regime


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config.yaml')
    p.add_argument('--top-n', type=int, default=5)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg['output']['report_dir'])
    model_dir = Path(cfg['output']['model_dir'])

    clf = BinaryDirectionClassifier.load(model_dir / 'classifier_model.pkl')
    vol_model = VolatilityModel.load(model_dir / 'vol_model.pkl')
    logger.info(f"Loaded classifier with {len(clf.feature_cols)} features")

    fetcher = DataFetcher(cfg['output']['data_cache_dir'])
    tickers = get_training_tickers()  # need full universe for feature context (sector RS etc.)
    trading = set(get_trading_tickers())
    end_date = datetime.now()
    start_date = end_date - timedelta(days=400)

    logger.info(f"Fetching latest data for {len(tickers)} tickers...")
    ohlcv = fetcher.get_ohlcv_batch(tickers, start_date, end_date, progress=False)
    macro = {}
    for m in [cfg['universe']['benchmark']] + cfg['universe']['sector_etfs'] + cfg['universe']['macro_tickers']:
        df = fetcher.get_ohlcv(m, start_date, end_date)
        if not df.empty:
            macro[m] = df
    fundamentals = {t: fetcher.get_fundamentals(t) for t in ohlcv.keys()}
    earnings = {t: fetcher.get_next_earnings_date(t) for t in ohlcv.keys()}

    fb = FeatureBuilder(
        return_windows=tuple(cfg['features']['return_windows']),
        vol_windows=tuple(cfg['features']['vol_windows']),
        target_horizon=cfg['target']['horizon_days'],
        target_top_pct=cfg['target']['top_pct'],
        sector_map=get_training_sector_map(),
    )
    panel = fb.build_panel(ohlcv, fundamentals, macro, earnings, cfg['universe']['benchmark'])
    panel, regime_info = merge_regime(panel, macro)
    panel['date'] = pd.to_datetime(panel['date'])

    latest_date = panel['date'].max()
    latest = panel[panel['date'] == latest_date].copy()
    # Filter to trading universe
    latest = latest[latest['ticker'].isin(trading)].copy()

    probs = clf.predict_proba(latest)
    vol_preds = vol_model.predict(latest)
    latest['score'] = probs.values            # P(up_3pct_21d)
    latest['p_up'] = probs.values
    latest['predicted_vol'] = vol_preds.values
    latest['score_pct'] = latest['score'].rank(pct=True)
    latest['liquidity_tier'] = latest['ticker'].map(get_liquidity_tier)
    latest['spot'] = latest['close']

    logger.info(f"Regime: {regime_info.get('regime')} (score {regime_info.get('regime_score', 0):.2f})")

    picks = latest.sort_values('score', ascending=False).head(args.top_n)
    picks = picks[['ticker', 'spot', 'score', 'p_up', 'score_pct', 'sector',
                   'predicted_vol', 'liquidity_tier']].reset_index(drop=True)

    # Kelly sizing using P(up) as edge proxy
    kelly = KellySizer()
    edges = (picks.set_index('ticker')['p_up'] - 0.5) * 0.10  # convert prob to expected edge
    edges = edges.clip(lower=0)
    vols = picks.set_index('ticker')['predicted_vol']
    weights = kelly.size(edges, vols, regime_score=regime_info.get('regime_score', 0))
    picks['kelly_weight'] = picks['ticker'].map(weights).fillna(0.0)
    port_stats = kelly.expected_portfolio_stats(weights, edges, vols)

    # Options strategy per pick
    selector = StrategySelector(
        expiry_dte_short=cfg['options'].get('expiry_dte', 30),
        iv_to_rv_ratio=cfg['options']['iv_to_rv_ratio'],
        risk_per_trade=cfg['options']['risk_per_trade'],
    )
    options_trades = []
    for _, row in picks.iterrows():
        df = ohlcv.get(row['ticker'])
        rv21 = 0.30
        if df is not None and len(df) > 21:
            rv21 = float(np.log(df['Close']).diff().tail(21).std() * (252 ** 0.5))
        ed = earnings.get(row['ticker'])
        days_to_earn = None
        if ed is not None:
            days_to_earn = (pd.Timestamp(ed) - pd.Timestamp.now()).days
        trade = selector.select(
            ticker=row['ticker'], spot=row['spot'],
            direction_prob=row['p_up'], predicted_vol=row['predicted_vol'],
            current_rv_21d=rv21, liquidity_tier=row['liquidity_tier'],
            days_to_earnings=days_to_earn, regime_score=regime_info.get('regime_score', 0),
        )
        options_trades.append(trade.to_dict())

    report_path = write_daily_report(
        picks, options_trades, out_dir, latest_date,
        regime_info=regime_info, portfolio_stats=port_stats,
    )
    logger.info(f"Daily picks: {report_path}")
    print("\nTop picks today:")
    print(picks.to_string(index=False))


if __name__ == '__main__':
    main()
