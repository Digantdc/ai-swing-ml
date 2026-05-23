"""Curated v2 feature set — 33 features that passed the IC analysis.

Derived from empirical analysis on 1485 trading days × 60 AI tickers:
    output/feature_ic_ranking.csv → top 33 by composite (AUC + IC) score.

Rationale per family:

  MACRO / REGIME (7):
    The single strongest predictor family. Yield curve slope at IC 0.116,
    VIX at 0.081 — these dominate the entire ranking. They predict MARKET
    direction, which gates whether to take stock-specific positions.

  FUNDAMENTAL QUALITY (13):
    Stock-specific edge. ROE, ROA, profit margin tell you which stocks
    structurally outperform. EV/EBITDA, P/B, P/S, PEG provide valuation
    anchoring. EPS growth catches accelerating businesses.

  LONG-HORIZON MOMENTUM (3):
    The ONE technical-like signal that survived: position within 52-week range
    + 252-day drawdown + 120-day return. These are momentum/cycle indicators,
    not short-term oscillators.

  VOLATILITY REGIME (3):
    Low-vol anomaly is real. High realized vol predicts LOWER forward returns.

  SENTIMENT / FLOW (4):
    Short interest, analyst target upside, analyst recommendations — these
    don't blow doors but contribute consistent IC.

  LONG-WINDOW TECHNICALS (3):
    Only the multi-week-timescale technicals survived. ADX (trend strength),
    SMA200 distance (regime indicator), ATR percent (vol normalization).

DELIBERATELY EXCLUDED (per IC analysis — all had AUC ~0.50):
    RSI(14), RSI(5), MACD, MACD histogram, MACD above signal,
    Stochastic K/D/diff, Bollinger %B, Bollinger width, BB width,
    SMA20/50 distance, EMA9/21 distance, SMA stack, Golden cross,
    DI difference, VWAP distance, OBV slope, MFI, volume 5-vs-20,
    volume z-score, dollar volume z-score, up/down volume ratio,
    Short returns (1d, 5d, 10d, 20d return + z-scores),
    Calendar features (day of week, OPEX, FOMC within 5, earnings within X),
    Sector ETF cross-references at short horizons (SMH, SOXX 21d returns),
    px_gap, px_range_ratio, px_streak.

These 50+ features are kept in feature engineering for backwards compatibility
but NOT passed to the v2 classifier.
"""

# ============================================================================
# Curated 33-feature set for v2 XGBoost binary classifier
# ============================================================================

V2_FEATURES = {
    # ----- Macro / Regime (7) -----
    'macro': [
        'mc_yield_curve_slope',      # rank 1 — IC +0.116, AUC 0.553
        'mc_vix',                    # rank 2 — IC +0.081, AUC 0.546
        'mc_spy_above_200ma',        # rank 4 — IC -0.077, AUC 0.465 (inverse)
        'mc_igv_rel_spy_21d',        # rank 6 — IC +0.069
        'mc_spy_rv_21d',             # rank 10 — IC +0.051
        'mc_spy_ret_21d',            # rank 13 — IC -0.057
        'mc_vix_zscore_60d',         # supplementary VIX dynamics
    ],

    # ----- Fundamental quality (13) -----
    'fundamental': [
        'fnd_roe',                   # rank 3 — IC +0.078
        'fnd_profit_margin',         # rank 5 — IC +0.072
        'fnd_roa',                   # rank 7 — IC +0.068
        'fnd_ev_ebitda',             # rank 8 — IC +0.065
        'fnd_pb',                    # rank 9 — IC +0.065
        'fnd_op_margin',             # rank 11 — IC +0.061
        'fnd_eps_growth_y',          # rank 15 — IC +0.053
        'fnd_eps_growth_q',          # rank 16 — IC +0.049
        'fnd_ev_rev',                # rank 18 — IC +0.046
        'fnd_pe_forward',            # rank 19 — IC +0.043
        'fnd_ps',                    # rank 20 — IC +0.045
        'fnd_fcf_margin',            # IC +0.029
        'fnd_gross_margin',          # IC +0.026
    ],

    # ----- Long-horizon momentum / position (3) -----
    'momentum_long': [
        'fnd_pct_of_52w_range',      # rank 12 — IC +0.061 — only "technical" that works
        'px_drawdown_252d',          # 252-day drawdown — IC +0.042
        'px_ret_120d',               # 120-day return — moderate signal
    ],

    # ----- Volatility regime (3) -----
    'volatility': [
        'vol_rv_zscore_252d',        # rank 14 — IC -0.055 — low-vol anomaly
        'vol_rv_60d',                # 60-day realized vol
        'vol_parkinson_21d',         # high-low based vol estimator
    ],

    # ----- Sentiment / flow (4) -----
    'sentiment': [
        'fnd_target_upside',         # rank 17 — IC -0.049 — analyst expectations
        'fnd_short_pct_float',       # short interest as crowdedness proxy
        'fnd_analyst_mean',          # consensus rating
        'fnd_n_analysts',            # coverage breadth
    ],

    # ----- Long-window technicals only (3) -----
    'technicals_long': [
        'ta_adx',                    # trend strength — only TA passing
        'ta_sma_200_dist',           # distance from 200-SMA (regime indicator)
        'ta_atr_pct',                # ATR normalized — vol-context only
    ],
}


def get_v2_feature_list() -> list[str]:
    """Flatten the curated dict into a single list."""
    out = []
    for family in V2_FEATURES.values():
        out.extend(family)
    return out


def get_v2_features_by_family() -> dict[str, list[str]]:
    return dict(V2_FEATURES)


# ============================================================================
# Helper: filter a feature panel to only v2-approved features
# ============================================================================

def filter_to_v2(panel_columns: list[str]) -> list[str]:
    """Return the subset of panel columns that are v2-approved.

    Useful when the FeatureBuilder produces ~108 features and we only want
    to feed 33 to the classifier.
    """
    v2 = set(get_v2_feature_list())
    return [c for c in panel_columns if c in v2]


# ============================================================================
# Regime-broadcast features added to the panel for v2
# ============================================================================

REGIME_BROADCAST_FEATURES = [
    'regime_score',               # -1 to +1 from RegimeDetector
    'regime_label_encoded',       # 0-4 ordinal
]


def get_v2_feature_list_with_regime() -> list[str]:
    """v2 feature list + regime broadcast features."""
    return get_v2_feature_list() + REGIME_BROADCAST_FEATURES
