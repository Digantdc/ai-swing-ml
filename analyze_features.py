#!/usr/bin/env python3
"""
analyze_features.py — Empirical feature predictivity analysis.

Place this file in ~/Documents/ai_swing_ml/ (alongside run_backtest.py).
Run from inside that folder with the venv active:

    cd ~/Documents/ai_swing_ml
    source .venv/bin/activate
    python analyze_features.py

What it does:
    1. Loads your existing OHLCV / fundamentals / macro panel (uses the same
       FeatureBuilder pipeline as the backtest — no re-fetching needed).
    2. Computes 21-day forward log return as the target.
    3. For EVERY feature, computes:
        - Pearson correlation with target
        - Spearman rank correlation (the gold-standard "Information Coefficient")
        - Mutual Information (catches non-linear relationships)
        - IC stability (rolling 6-month IC std)
        - Regime-conditional IC (bull / chop / risk-off separately)
    4. Outputs:
        - output/feature_ic_ranking.csv   — full ranking, sortable
        - output/feature_ic_report.md     — top-30 summary + interpretation
        - output/feature_ic_by_regime.csv — regime-conditional IC matrix
    5. Verdict: which features pass the IC > 0.02 threshold for v2 inclusion.
"""
from __future__ import annotations

import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('feature_analysis')

# ---------------------------------------------------------------- imports

try:
    from scipy import stats
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.metrics import roc_auc_score
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install: pip install scipy scikit-learn")
    sys.exit(1)

from data.fetcher import DataFetcher
from data.universe import UNIVERSE
from features.builder import FeatureBuilder
from models.regime_detector import RegimeDetector


# ============================================================================
# IC computation functions
# ============================================================================

def cross_sectional_ic(
    df: pd.DataFrame,
    feature_col: str,
    target_col: str,
    method: str = 'spearman',
) -> pd.Series:
    """Compute IC per date (cross-sectional), then return the series of daily ICs.

    Standard quant practice: IC is computed cross-sectionally each day
    (across stocks on the same date), then averaged. This is the right
    metric for relative ranking power.
    """
    daily_ic = []
    dates = []
    for date, group in df.groupby('date'):
        sub = group[[feature_col, target_col]].dropna()
        if len(sub) < 5:
            continue
        if sub[feature_col].std() == 0 or sub[target_col].std() == 0:
            continue
        if method == 'spearman':
            ic, _ = stats.spearmanr(sub[feature_col], sub[target_col])
        else:
            ic, _ = stats.pearsonr(sub[feature_col], sub[target_col])
        if not np.isnan(ic):
            daily_ic.append(ic)
            dates.append(date)
    return pd.Series(daily_ic, index=pd.DatetimeIndex(dates), name=feature_col)


def time_series_ic(
    df: pd.DataFrame,
    feature_col: str,
    target_col: str,
    method: str = 'spearman',
) -> float:
    """Single IC across all (date, ticker) rows — overall predictive strength."""
    sub = df[[feature_col, target_col]].dropna()
    if len(sub) < 100 or sub[feature_col].std() == 0:
        return np.nan
    if method == 'spearman':
        ic, _ = stats.spearmanr(sub[feature_col], sub[target_col])
    else:
        ic, _ = stats.pearsonr(sub[feature_col], sub[target_col])
    return ic


def mutual_info(df: pd.DataFrame, feature_col: str, target_col: str, n_sample: int = 5000) -> float:
    """Mutual information — captures non-linear relationships Pearson misses."""
    sub = df[[feature_col, target_col]].dropna()
    if len(sub) < 100:
        return np.nan
    if len(sub) > n_sample:
        sub = sub.sample(n_sample, random_state=42)
    try:
        mi = mutual_info_regression(
            sub[[feature_col]].values,
            sub[target_col].values,
            random_state=42,
        )[0]
        return mi
    except Exception:
        return np.nan


def binary_auc(df: pd.DataFrame, feature_col: str, target_col: str) -> float:
    """AUC for a single feature vs a binary target.

    AUC = probability that a random positive ranks higher than a random negative.
    0.50 = random; 0.60+ = useful for classification; 0.70+ = suspect (check leakage).
    """
    sub = df[[feature_col, target_col]].dropna()
    if len(sub) < 100:
        return np.nan
    if sub[feature_col].std() == 0:
        return np.nan
    n_pos = sub[target_col].sum()
    n_neg = len(sub) - n_pos
    if n_pos < 10 or n_neg < 10:
        return np.nan
    try:
        return float(roc_auc_score(sub[target_col].values, sub[feature_col].values))
    except Exception:
        return np.nan


def information_value(df: pd.DataFrame, feature_col: str, target_col: str, n_bins: int = 10) -> float:
    """Information Value — credit-scoring standard for binary feature evaluation.

    IV < 0.02: useless
    IV 0.02-0.10: weak predictor
    IV 0.10-0.30: medium predictor
    IV 0.30+: strong predictor (rare; check leakage if > 0.5)
    """
    sub = df[[feature_col, target_col]].dropna()
    if len(sub) < 200 or sub[feature_col].nunique() < n_bins:
        return np.nan
    try:
        sub = sub.copy()
        sub['bin'] = pd.qcut(sub[feature_col], q=n_bins, duplicates='drop')
        total_pos = sub[target_col].sum()
        total_neg = len(sub) - total_pos
        if total_pos == 0 or total_neg == 0:
            return np.nan
        iv = 0.0
        for _, group in sub.groupby('bin', observed=True):
            pos = group[target_col].sum()
            neg = len(group) - pos
            # Apply Laplace smoothing to avoid log(0)
            pos_rate = max(pos, 0.5) / total_pos
            neg_rate = max(neg, 0.5) / total_neg
            woe = np.log(pos_rate / neg_rate)
            iv += (pos_rate - neg_rate) * woe
        return float(iv)
    except Exception:
        return np.nan


def regime_conditional_ic(
    df: pd.DataFrame,
    feature_col: str,
    target_col: str,
    regime_col: str = 'regime_score',
) -> dict[str, float]:
    """IC computed separately for bull / chop / risk-off regimes."""
    out = {}
    if regime_col not in df.columns:
        return out
    # Bull: regime_score > 0.5
    # Chop: -0.5 <= regime_score <= 0.5
    # Risk-off: regime_score < -0.5
    masks = {
        'bull': df[regime_col] > 0.5,
        'chop': (df[regime_col] >= -0.5) & (df[regime_col] <= 0.5),
        'risk_off': df[regime_col] < -0.5,
    }
    for label, mask in masks.items():
        sub = df[mask][[feature_col, target_col]].dropna()
        if len(sub) < 50:
            out[label] = np.nan
            continue
        try:
            ic, _ = stats.spearmanr(sub[feature_col], sub[target_col])
            out[label] = ic
        except Exception:
            out[label] = np.nan
    return out


# ============================================================================
# Main pipeline
# ============================================================================

def build_panel_with_target(years: int = 5) -> pd.DataFrame:
    """Reuse the v1 pipeline to build features + 21d forward return target."""
    with open('config.yaml') as f:
        cfg = yaml.safe_load(f)

    cache_dir = Path(cfg['output']['data_cache_dir'])
    fetcher = DataFetcher(cache_dir)
    tickers = list(UNIVERSE.keys())
    end_date = datetime.now()
    start_date = end_date - timedelta(days=int(365 * (years + 1)))

    logger.info(f"Loading OHLCV for {len(tickers)} tickers (uses cache where possible)...")
    ohlcv = fetcher.get_ohlcv_batch(tickers, start_date, end_date, progress=False)
    logger.info(f"  loaded {len(ohlcv)} tickers")

    macro_list = [cfg['universe']['benchmark']] + cfg['universe']['sector_etfs'] + cfg['universe']['macro_tickers']
    macro_panel = {}
    for m in macro_list:
        df = fetcher.get_ohlcv(m, start_date, end_date)
        if not df.empty:
            macro_panel[m] = df

    logger.info("Loading fundamentals...")
    fundamentals = {t: fetcher.get_fundamentals(t) for t in ohlcv.keys()}
    earnings_dates = {t: fetcher.get_next_earnings_date(t) for t in ohlcv.keys()}

    sector_map = {t: UNIVERSE[t][0] for t in ohlcv.keys() if t in UNIVERSE}
    fb = FeatureBuilder(
        return_windows=tuple(cfg['features']['return_windows']),
        vol_windows=tuple(cfg['features']['vol_windows']),
        target_horizon=cfg['target']['horizon_days'],
        target_top_pct=cfg['target']['top_pct'],
        sector_map=sector_map,
    )
    logger.info("Building features...")
    panel = fb.build_panel(ohlcv, fundamentals, macro_panel, earnings_dates, cfg['universe']['benchmark'])
    panel = fb.add_targets(panel)
    logger.info(f"  panel shape: {panel.shape}")

    # Attach regime score
    logger.info("Detecting regimes...")
    rd = RegimeDetector()
    regime_df = rd.detect(macro_panel)
    if not regime_df.empty:
        regime_df = regime_df.reset_index()
        regime_df['date'] = pd.to_datetime(regime_df['date'])
        panel['date'] = pd.to_datetime(panel['date'])
        panel = panel.merge(
            regime_df[['date', 'regime_score', 'regime']],
            on='date', how='left',
        )

    return panel


def analyze_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Run all predictivity tests for every feature against MULTIPLE targets:
        - target_fwd_return (continuous 21d log return) — for magnitude prediction
        - target_up_in_21d   (binary: return > 0)        — for direction prediction
        - target_up_3pct     (binary: return > 3%)       — for tradable threshold

    The continuous target is what's typically used in quant feature research.
    The binary targets are what the XGBoost classifier in v2 actually predicts.
    Reporting both side-by-side lets you see which features are good for
    classification (binary AUC) vs which are good for sizing (Spearman IC).
    """
    panel = panel.copy()
    target_cont = 'target_fwd_return'
    target_bin_any = 'target_up_in_21d'
    target_bin_3pct = 'target_up_3pct'
    # Build binary targets from the continuous one
    panel[target_bin_any] = (panel[target_cont] > 0).astype(int)
    panel[target_bin_3pct] = (panel[target_cont] > 0.03).astype(int)

    # Identify feature columns (exclude meta + targets)
    meta_cols = {
        'ticker', 'date', 'close', 'sector',
        'target_fwd_return', 'target_fwd_vol',
        'target_rank_pct', 'target_top_pct', 'target_relevance',
        'target_up_in_21d', 'target_up_3pct',
        'regime', 'regime_score',
    }
    feature_cols = [c for c in panel.columns if c not in meta_cols]
    logger.info(f"Analyzing {len(feature_cols)} features against 3 targets (continuous + 2 binary)...")

    results = []
    for i, feat in enumerate(feature_cols, 1):
        if i % 20 == 0:
            logger.info(f"  progress: {i}/{len(feature_cols)}")
        # Skip non-numeric
        if not pd.api.types.is_numeric_dtype(panel[feat]):
            continue

        try:
            row = {'feature': feat}

            # === CONTINUOUS TARGET: magnitude prediction ===
            row['pearson_overall'] = time_series_ic(panel, feat, target_cont, 'pearson')
            row['spearman_overall'] = time_series_ic(panel, feat, target_cont, 'spearman')

            # Cross-sectional IC against continuous target
            daily_ic_spear = cross_sectional_ic(panel, feat, target_cont, 'spearman')
            row['spearman_xs_mean'] = daily_ic_spear.mean() if len(daily_ic_spear) > 0 else np.nan
            row['spearman_xs_std'] = daily_ic_spear.std() if len(daily_ic_spear) > 0 else np.nan
            if row['spearman_xs_std'] and row['spearman_xs_std'] > 0:
                row['ic_information_ratio'] = row['spearman_xs_mean'] / row['spearman_xs_std']
            else:
                row['ic_information_ratio'] = np.nan
            row['n_dates'] = len(daily_ic_spear)

            # === BINARY TARGETS: direction prediction (what classifier cares about) ===
            row['auc_up_any'] = binary_auc(panel, feat, target_bin_any)
            row['auc_up_3pct'] = binary_auc(panel, feat, target_bin_3pct)
            row['info_value_3pct'] = information_value(panel, feat, target_bin_3pct)

            # Mutual info (against continuous — robust to non-linearity)
            row['mutual_info'] = mutual_info(panel, feat, target_cont)

            # Regime-conditional IC (against continuous)
            regime_ic = regime_conditional_ic(panel, feat, target_cont)
            for label in ('bull', 'chop', 'risk_off'):
                row[f'spearman_{label}'] = regime_ic.get(label, np.nan)
            regime_values = [regime_ic.get(l) for l in ('bull', 'chop', 'risk_off') if not np.isnan(regime_ic.get(l, np.nan))]
            if len(regime_values) >= 2:
                row['regime_ic_range'] = max(regime_values) - min(regime_values)
            else:
                row['regime_ic_range'] = np.nan

            results.append(row)
        except Exception as e:
            logger.warning(f"  failed for {feat}: {e}")
            continue

    df = pd.DataFrame(results)
    # Composite score: combines magnitude IC (continuous) + direction AUC (binary)
    # Maps both to comparable scales: |IC| × 2 + (AUC - 0.5) × 4
    # Magnitude weight = direction weight, so features good at both rank highest
    df['abs_ic'] = df['spearman_overall'].abs().fillna(0)
    df['abs_auc_edge'] = (df['auc_up_3pct'].fillna(0.5) - 0.5).abs()
    df['composite_score'] = df['abs_ic'] * 2 + df['abs_auc_edge'] * 4
    df = df.sort_values('composite_score', ascending=False).drop(columns=['abs_ic', 'abs_auc_edge'])
    return df


def write_report(ic_df: pd.DataFrame, out_dir: Path):
    """Write the markdown summary + CSVs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # Full ranking CSV
    ic_df.to_csv(out_dir / 'feature_ic_ranking.csv', index=False)

    # Regime-by-feature matrix
    regime_cols = ['feature', 'spearman_overall', 'spearman_bull', 'spearman_chop', 'spearman_risk_off', 'regime_ic_range']
    ic_df[regime_cols].to_csv(out_dir / 'feature_ic_by_regime.csv', index=False)

    # Markdown report
    lines = []
    lines.append("# Feature Predictivity Analysis — IC + AUC Report\n")
    lines.append(f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    lines.append(f"Total features analyzed: **{len(ic_df)}**\n")

    lines.append("## What each metric means\n")
    lines.append("- **Spearman IC** (against continuous 21d return) — rank correlation; |IC| ≥ 0.02 is useful")
    lines.append("- **AUC_up_3pct** (against binary: return > 3%) — classifier discrimination; 0.55+ is useful, 0.60+ is good")
    lines.append("- **Info Value** (against binary) — credit-scoring metric; 0.1+ medium, 0.3+ strong")
    lines.append("- **IC stability** (mean/std of cross-sectional ICs) — higher = more consistent")
    lines.append("- **Composite score** — `|IC|·2 + (AUC-0.5)·4`. Combines magnitude + direction predictivity.\n")

    # Threshold counts based on AUC (the binary classifier target)
    strong_auc = ic_df[ic_df['auc_up_3pct'].fillna(0.5).sub(0.5).abs() >= 0.05]   # AUC > 0.55 or < 0.45
    moderate_auc = ic_df[(ic_df['auc_up_3pct'].fillna(0.5).sub(0.5).abs() >= 0.02) & (ic_df['auc_up_3pct'].fillna(0.5).sub(0.5).abs() < 0.05)]
    weak_auc = ic_df[(ic_df['auc_up_3pct'].fillna(0.5).sub(0.5).abs() >= 0.01) & (ic_df['auc_up_3pct'].fillna(0.5).sub(0.5).abs() < 0.02)]
    noise_auc = ic_df[ic_df['auc_up_3pct'].fillna(0.5).sub(0.5).abs() < 0.01]

    strong_ic = ic_df[ic_df['spearman_overall'].abs() >= 0.05]
    moderate_ic = ic_df[(ic_df['spearman_overall'].abs() >= 0.02) & (ic_df['spearman_overall'].abs() < 0.05)]
    weak_ic = ic_df[(ic_df['spearman_overall'].abs() >= 0.005) & (ic_df['spearman_overall'].abs() < 0.02)]
    noise_ic = ic_df[ic_df['spearman_overall'].abs() < 0.005]

    lines.append("## Distribution\n")
    lines.append("### By binary AUC (direction prediction — what the classifier optimizes)\n")
    lines.append(f"- **Strong** (|AUC−0.5| ≥ 0.05): **{len(strong_auc)}** features ← gold for the classifier")
    lines.append(f"- **Moderate** (0.02 ≤ |AUC−0.5| < 0.05): **{len(moderate_auc)}** features ← useful")
    lines.append(f"- **Weak** (0.01 ≤ |AUC−0.5| < 0.02): **{len(weak_auc)}** features ← marginal")
    lines.append(f"- **Noise** (|AUC−0.5| < 0.01): **{len(noise_auc)}** features ← drop for the classifier\n")
    lines.append("### By continuous IC (magnitude prediction — useful for sizing)\n")
    lines.append(f"- **Strong** (|IC| ≥ 0.05): **{len(strong_ic)}** features")
    lines.append(f"- **Moderate** (0.02 ≤ |IC| < 0.05): **{len(moderate_ic)}** features")
    lines.append(f"- **Weak** (0.005 ≤ |IC| < 0.02): **{len(weak_ic)}** features")
    lines.append(f"- **Noise** (|IC| < 0.005): **{len(noise_ic)}** features\n")

    # Top 30 by composite (combines magnitude + direction)
    lines.append("## Top 30 features by composite score (magnitude + direction)\n")
    lines.append("| Rank | Feature | Spearman IC | AUC_up_3pct | Info Value | IC stability | Mutual Info | Bull IC | Chop IC | Risk-off IC |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    top = ic_df.head(30)
    for i, row in enumerate(top.itertuples(index=False), 1):
        def fmt(v, dp=3):
            if pd.isna(v):
                return 'n/a'
            return f"{v:+.{dp}f}"
        def fmt_auc(v):
            if pd.isna(v):
                return 'n/a'
            return f"{v:.3f}"
        lines.append(
            f"| {i} | `{row.feature}` "
            f"| {fmt(row.spearman_overall)} "
            f"| {fmt_auc(row.auc_up_3pct)} "
            f"| {fmt(row.info_value_3pct, 3)} "
            f"| {fmt(row.ic_information_ratio, 2)} "
            f"| {fmt(row.mutual_info)} "
            f"| {fmt(row.spearman_bull)} "
            f"| {fmt(row.spearman_chop)} "
            f"| {fmt(row.spearman_risk_off)} |"
        )

    # Notable features by category
    lines.append("\n## How your stated technical indicators rank\n")
    lines.append("These are the indicators you trade on. We tested each against both "
                 "the continuous return (magnitude) and the binary up/down target (direction).\n")
    user_techs = ['ta_rsi_14', 'ta_macd', 'ta_macd_hist', 'ta_bb_pct', 'ta_bb_width',
                  'ta_atr_pct', 'ta_sma_20_dist', 'ta_sma_50_dist', 'ta_sma_200_dist',
                  'ta_stoch_k', 'ta_stoch_d', 'vol_vwap_dist', 'vol_obv_slope_20',
                  'vol_5_vs_20', 'vol_mfi_14']
    lines.append("| Feature | Spearman IC | AUC_up_3pct | Composite Rank | Tier (binary) |")
    lines.append("|---|---|---|---|---|")
    ic_df_indexed = ic_df.reset_index(drop=True)
    for tech in user_techs:
        match = ic_df_indexed[ic_df_indexed['feature'] == tech]
        if match.empty:
            lines.append(f"| `{tech}` | not found | n/a | — | — |")
            continue
        row = match.iloc[0]
        ic = row['spearman_overall']
        auc = row['auc_up_3pct']
        rank_pos = int(match.index[0]) + 1
        # Tier by AUC (since the model is binary)
        if pd.isna(auc):
            tier = 'n/a'
        else:
            auc_edge = abs(auc - 0.5)
            if auc_edge >= 0.05:
                tier = 'A (strong)'
            elif auc_edge >= 0.02:
                tier = 'B (moderate)'
            elif auc_edge >= 0.01:
                tier = 'C (weak)'
            else:
                tier = 'D (noise)'
        ic_str = f"{ic:+.3f}" if not pd.isna(ic) else 'n/a'
        auc_str = f"{auc:.3f}" if not pd.isna(auc) else 'n/a'
        lines.append(f"| `{tech}` | {ic_str} | {auc_str} | #{rank_pos} | {tier} |")

    # Features to consider DROPPING — noise in BOTH binary AUC and continuous IC
    noise_in_both = ic_df[
        (ic_df['auc_up_3pct'].fillna(0.5).sub(0.5).abs() < 0.01)
        & (ic_df['spearman_overall'].abs() < 0.005)
    ]
    lines.append("\n## Features to drop (noise in BOTH AUC and IC — useless for both classifier and vol model)\n")
    if len(noise_in_both) > 0:
        drop_list = noise_in_both['feature'].tolist()
        lines.append("These features contribute zero signal to either model. Remove from v2:\n")
        for f in drop_list:
            lines.append(f"- `{f}`")
    else:
        lines.append("All features carry at least minimal signal — none to drop.")

    # Verdict
    lines.append("\n## Verdict for v2\n")
    if len(strong_auc) > 0:
        lines.append(f"✅ **{len(strong_auc)} strong predictors by AUC** ← core inputs for the XGBoost binary classifier.")
    if len(moderate_auc) >= 10:
        lines.append(f"✅ **{len(moderate_auc)} moderate predictors by AUC** — healthy candidate pool for the classifier.")
    elif len(moderate_auc) < 5:
        lines.append(f"⚠️ Only **{len(moderate_auc)} moderate predictors by AUC** found — classifier feature set is thin.")
    if len(strong_ic) > 0:
        lines.append(f"✅ **{len(strong_ic)} strong predictors by IC** ← core inputs for the volatility/magnitude model.")
    if len(noise_in_both) > 30:
        lines.append(f"⚠️ **{len(noise_in_both)} features are noise in both metrics** — drop these to reduce overfit risk.")

    lines.append("\n**Recommended v2 inputs for the XGBoost binary classifier:**")
    lines.append("- Primary criterion: `|AUC − 0.5| ≥ 0.02` (B tier or better)")
    lines.append("- Stability filter: `IC information ratio ≥ 0.05`")
    lines.append("- Regime guard: `|regime IC range| ≤ 0.05` (feature doesn't flip sign by regime)")
    lines.append("\n**For the volatility forecaster** (separate model for magnitude/sizing):")
    lines.append("- Use features with `|Spearman IC| ≥ 0.02` against the continuous target.")
    lines.append("- These may differ from the classifier features — that's expected and correct.")

    (out_dir / 'feature_ic_report.md').write_text('\n'.join(lines))


def main():
    out_dir = Path('output')
    logger.info("=== Feature predictivity analysis ===")

    panel = build_panel_with_target()
    if panel.empty:
        logger.error("Empty panel — fix data fetch issues first.")
        sys.exit(1)

    # Drop rows with NaN target
    panel = panel.dropna(subset=['target_fwd_return'])
    logger.info(f"Panel after target filter: {panel.shape}")

    ic_df = analyze_features(panel)
    logger.info(f"Computed IC for {len(ic_df)} features")

    write_report(ic_df, out_dir)
    logger.info(f"Report written to {out_dir / 'feature_ic_report.md'}")
    logger.info(f"Full ranking: {out_dir / 'feature_ic_ranking.csv'}")
    logger.info(f"Regime matrix: {out_dir / 'feature_ic_by_regime.csv'}")

    # Print top 10 to console
    print("\n=== TOP 10 BY ABSOLUTE IC ===")
    print(ic_df[['feature', 'spearman_overall', 'spearman_xs_mean', 'mutual_info']].head(10).to_string(index=False))


if __name__ == '__main__':
    main()
