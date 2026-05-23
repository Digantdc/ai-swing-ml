# Feature Predictivity Analysis — IC + AUC Report

Generated 2026-05-20 02:46

Total features analyzed: **104**

## What each metric means

- **Spearman IC** (against continuous 21d return) — rank correlation; |IC| ≥ 0.02 is useful
- **AUC_up_3pct** (against binary: return > 3%) — classifier discrimination; 0.55+ is useful, 0.60+ is good
- **Info Value** (against binary) — credit-scoring metric; 0.1+ medium, 0.3+ strong
- **IC stability** (mean/std of cross-sectional ICs) — higher = more consistent
- **Composite score** — `|IC|·2 + (AUC-0.5)·4`. Combines magnitude + direction predictivity.

## Distribution

### By binary AUC (direction prediction — what the classifier optimizes)

- **Strong** (|AUC−0.5| ≥ 0.05): **1** features ← gold for the classifier
- **Moderate** (0.02 ≤ |AUC−0.5| < 0.05): **20** features ← useful
- **Weak** (0.01 ≤ |AUC−0.5| < 0.02): **24** features ← marginal
- **Noise** (|AUC−0.5| < 0.01): **59** features ← drop for the classifier

### By continuous IC (magnitude prediction — useful for sizing)

- **Strong** (|IC| ≥ 0.05): **15** features
- **Moderate** (0.02 ≤ |IC| < 0.05): **33** features
- **Weak** (0.005 ≤ |IC| < 0.02): **40** features
- **Noise** (|IC| < 0.005): **16** features

## Top 30 features by composite score (magnitude + direction)

| Rank | Feature | Spearman IC | AUC_up_3pct | Info Value | IC stability | Mutual Info | Bull IC | Chop IC | Risk-off IC |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `mc_yield_curve_slope` | +0.116 | 0.553 | +0.085 | n/a | +0.045 | +0.114 | +0.134 | +0.218 |
| 2 | `mc_vix` | +0.081 | 0.546 | +0.050 | n/a | +0.038 | +0.040 | +0.075 | +0.153 |
| 3 | `fnd_roe` | +0.078 | 0.536 | +0.020 | +0.42 | +0.068 | +0.089 | +0.080 | +0.035 |
| 4 | `mc_spy_above_200ma` | -0.077 | 0.465 | n/a | n/a | +0.000 | -0.064 | -0.047 | n/a |
| 5 | `fnd_profit_margin` | +0.072 | 0.532 | +0.019 | +0.43 | +0.053 | +0.075 | +0.080 | +0.030 |
| 6 | `mc_igv_rel_spy_21d` | +0.069 | 0.533 | +0.023 | n/a | +0.026 | -0.006 | +0.073 | +0.405 |
| 7 | `fnd_roa` | +0.068 | 0.531 | +0.015 | +0.39 | +0.060 | +0.073 | +0.076 | +0.021 |
| 8 | `fnd_ev_ebitda` | +0.065 | 0.530 | +0.016 | +0.44 | +0.060 | +0.053 | +0.076 | +0.063 |
| 9 | `fnd_pb` | +0.065 | 0.529 | +0.017 | +0.46 | +0.007 | +0.069 | +0.071 | +0.032 |
| 10 | `mc_spy_rv_21d` | +0.051 | 0.534 | +0.035 | n/a | +0.032 | +0.060 | +0.004 | +0.092 |
| 11 | `fnd_op_margin` | +0.061 | 0.527 | +0.016 | +0.38 | +0.040 | +0.062 | +0.073 | +0.014 |
| 12 | `fnd_pct_of_52w_range` | +0.061 | 0.527 | +0.018 | +0.34 | +0.045 | +0.061 | +0.072 | +0.025 |
| 13 | `mc_spy_ret_21d` | -0.057 | 0.473 | +0.013 | n/a | +0.016 | -0.020 | -0.018 | +0.135 |
| 14 | `vol_rv_zscore_252d` | -0.055 | 0.473 | +0.010 | -0.13 | +0.020 | -0.068 | -0.089 | +0.032 |
| 15 | `fnd_eps_growth_y` | +0.053 | 0.524 | +0.014 | +0.28 | +0.053 | +0.079 | +0.035 | +0.027 |
| 16 | `fnd_eps_growth_q` | +0.049 | 0.523 | +0.011 | +0.26 | +0.037 | +0.071 | +0.036 | +0.021 |
| 17 | `fnd_target_upside` | -0.049 | 0.478 | +0.018 | -0.27 | +0.032 | -0.036 | -0.070 | -0.022 |
| 18 | `fnd_ev_rev` | +0.046 | 0.521 | +0.007 | +0.31 | +0.004 | +0.069 | +0.024 | +0.042 |
| 19 | `fnd_pe_forward` | +0.043 | 0.523 | +0.013 | +0.33 | +0.056 | +0.024 | +0.065 | +0.034 |
| 20 | `fnd_ps` | +0.045 | 0.520 | +0.008 | +0.32 | +0.002 | +0.067 | +0.022 | +0.049 |
| 21 | `mc_soxx_ret_21d` | -0.042 | 0.479 | +0.014 | n/a | +0.006 | +0.055 | -0.049 | -0.045 |
| 22 | `fnd_short_pct_float` | -0.043 | 0.480 | +0.013 | -0.32 | +0.042 | -0.034 | -0.065 | +0.005 |
| 23 | `mc_smh_ret_21d` | -0.037 | 0.480 | +0.014 | n/a | +0.011 | +0.056 | -0.037 | +0.017 |
| 24 | `mc_dxy_ret_21d` | -0.035 | 0.480 | +0.009 | n/a | +0.016 | -0.083 | +0.005 | -0.137 |
| 25 | `mc_10y_change_21d` | +0.033 | 0.517 | +0.032 | n/a | +0.035 | -0.033 | +0.076 | -0.071 |
| 26 | `fnd_fcf_margin` | +0.029 | 0.517 | +0.006 | +0.23 | +0.023 | +0.010 | +0.050 | +0.025 |
| 27 | `fnd_n_analysts` | +0.029 | 0.515 | +0.011 | +0.19 | +0.048 | +0.024 | +0.045 | -0.010 |
| 28 | `mc_spy_ret_5d` | -0.030 | 0.487 | +0.008 | n/a | +0.006 | -0.021 | -0.013 | +0.046 |
| 29 | `px_drawdown_252d` | +0.042 | 0.507 | +0.006 | +0.21 | +0.043 | +0.059 | +0.090 | -0.026 |
| 30 | `fnd_gross_margin` | +0.026 | 0.515 | +0.006 | +0.17 | +0.055 | +0.019 | +0.036 | +0.008 |

## How your stated technical indicators rank

These are the indicators you trade on. We tested each against both the continuous return (magnitude) and the binary up/down target (direction).

| Feature | Spearman IC | AUC_up_3pct | Composite Rank | Tier (binary) |
|---|---|---|---|---|
| `ta_rsi_14` | -0.004 | 0.501 | #100 | D (noise) |
| `ta_macd` | -0.005 | 0.501 | #99 | D (noise) |
| `ta_macd_hist` | -0.007 | 0.500 | #92 | D (noise) |
| `ta_bb_pct` | -0.014 | 0.496 | #65 | D (noise) |
| `ta_bb_width` | -0.002 | 0.516 | #55 | C (weak) |
| `ta_atr_pct` | -0.016 | 0.510 | #53 | D (noise) |
| `ta_sma_20_dist` | -0.010 | 0.500 | #85 | D (noise) |
| `ta_sma_50_dist` | -0.016 | 0.497 | #66 | D (noise) |
| `ta_sma_200_dist` | +0.025 | 0.506 | #50 | D (noise) |
| `ta_stoch_k` | -0.005 | 0.502 | #95 | D (noise) |
| `ta_stoch_d` | -0.004 | 0.501 | #98 | D (noise) |
| `vol_vwap_dist` | -0.006 | 0.501 | #90 | D (noise) |
| `vol_obv_slope_20` | -0.013 | 0.498 | #72 | D (noise) |
| `vol_5_vs_20` | +0.029 | 0.511 | #34 | C (weak) |
| `vol_mfi_14` | +0.001 | 0.504 | #87 | D (noise) |

## Features to drop (noise in BOTH AUC and IC — useless for both classifier and vol model)

These features contribute zero signal to either model. Remove from v2:

- `rs_vs_bench_5d`
- `cal_dow`
- `rs_vs_bench_63d`
- `vol_dollar_zscore_60d`
- `vol_mfi_14`
- `fnd_peg_reasonable`
- `ta_stoch_k`
- `ta_rsi_5`
- `ta_stoch_d`
- `ta_macd`
- `ta_rsi_14`
- `vol_rv_5_vs_60`
- `ta_stoch_diff`
- `mc_igv_ret_21d`
- `cal_is_quarter_end_week`

## Verdict for v2

✅ **1 strong predictors by AUC** ← core inputs for the XGBoost binary classifier.
✅ **20 moderate predictors by AUC** — healthy candidate pool for the classifier.
✅ **15 strong predictors by IC** ← core inputs for the volatility/magnitude model.

**Recommended v2 inputs for the XGBoost binary classifier:**
- Primary criterion: `|AUC − 0.5| ≥ 0.02` (B tier or better)
- Stability filter: `IC information ratio ≥ 0.05`
- Regime guard: `|regime IC range| ≤ 0.05` (feature doesn't flip sign by regime)

**For the volatility forecaster** (separate model for magnitude/sizing):
- Use features with `|Spearman IC| ≥ 0.02` against the continuous target.
- These may differ from the classifier features — that's expected and correct.