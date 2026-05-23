#!/usr/bin/env python3
"""v2 production training — XGBoost binary classifier on full history.

Usage:
    python train.py --config config.yaml

Saves:
    output/classifier_model.pkl
    output/vol_model.pkl
    output/train_metadata.json
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml

from data.fetcher import DataFetcher
from data.training_universe import get_training_tickers, get_training_sector_map
from features.builder import FeatureBuilder
from features.feature_set import get_v2_feature_list_with_regime
from models.binary_classifier import BinaryDirectionClassifier
from models.volatility_model import VolatilityModel
from models.regime_detector import RegimeDetector

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger('train_v2')


def merge_regime(panel, macro_panel):
    rd = RegimeDetector()
    regime_df = rd.detect(macro_panel)
    if regime_df.empty:
        panel['regime_score'] = 0.0
        panel['regime_label_encoded'] = 2
        return panel
    regime_df = regime_df.reset_index()
    regime_df['date'] = pd.to_datetime(regime_df['date'])
    label_map = {'risk_off': 0, 'defensive': 1, 'chop': 2, 'cautious_bull': 3, 'trending_bull': 4}
    regime_df['regime_label_encoded'] = regime_df['regime'].map(label_map).fillna(2).astype(int)
    panel['date'] = pd.to_datetime(panel['date'])
    panel = panel.merge(regime_df[['date', 'regime_score', 'regime_label_encoded']], on='date', how='left')
    panel['regime_score'] = panel['regime_score'].fillna(0.0)
    panel['regime_label_encoded'] = panel['regime_label_encoded'].fillna(2).astype(int)
    return panel


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config.yaml')
    p.add_argument('--years', type=int, default=None)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    years = args.years or cfg['backtest']['total_years']
    fetcher = DataFetcher(cfg['output']['data_cache_dir'])
    out_dir = Path(cfg['output']['model_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    tickers = get_training_tickers()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=int(365 * (years + 1)))

    logger.info(f"Fetching {len(tickers)} training tickers, {years}y...")
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
    panel = fb.add_targets(panel)
    panel = merge_regime(panel, macro)
    panel['date'] = pd.to_datetime(panel['date'])

    feature_subset = [f for f in get_v2_feature_list_with_regime() if f in panel.columns]
    logger.info(f"Training with {len(feature_subset)} features")

    cutoff = panel['date'].max() - pd.DateOffset(months=3)
    train = panel[panel['date'] < cutoff].dropna(subset=['target_fwd_return'])
    valid = panel[panel['date'] >= cutoff].dropna(subset=['target_fwd_return'])
    logger.info(f"Train: {len(train)} rows, Valid: {len(valid)} rows")

    # Classifier
    clf = BinaryDirectionClassifier(
        params=cfg['model'].get('xgboost', {}),
        random_state=cfg['model'].get('random_state', 42),
        calibrate=True,
        feature_subset=feature_subset,
    )
    clf.fit(train, valid, target_threshold=cfg['target'].get('min_up_return', 0.03))
    clf.save(out_dir / 'classifier_model.pkl')
    logger.info(f"Classifier saved to {out_dir / 'classifier_model.pkl'}")

    # Vol model
    vm = VolatilityModel(**cfg['volatility_model']['rf'])
    vm.fit(train)
    vm.save(out_dir / 'vol_model.pkl')
    logger.info(f"Vol model saved to {out_dir / 'vol_model.pkl'}")

    meta = {
        'trained_at': datetime.now().isoformat(),
        'years': years, 'train_rows': len(train), 'valid_rows': len(valid),
        'n_features': len(feature_subset), 'features': feature_subset,
        'training_tickers': sorted(ohlcv.keys()),
    }
    with open(out_dir / 'train_metadata.json', 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    logger.info("Done.")


if __name__ == '__main__':
    main()
