# Backtest Report

_Generated 2026-05-21 13:47_

## Equity-leg performance

| Metric | Value |
|---|---|
| Total Return | 449.35% |
| Annualized Return | 59.14% |
| Annualized Volatility | 0.469 |
| Sharpe | 1.123 |
| Sortino | 3.696 |
| Max Drawdown | -25.76% |
| Calmar | 2.296 |
| N Trades | 220 |
| Hit Rate Trades | 54.55% |
| Avg Winner | 16.84% |
| Avg Loser | -9.72% |
| Profit Factor | 2.079 |
| Information Coefficient | 0.003 |
| Rank Ic | -0.021 |

## How to interpret these numbers

- **Sharpe 1.12** is decent for a retail strategy.
- **Rank IC -0.021** ≤ 0. No predictive power found.

## Top 20 features by importance

| Feature | Importance |
|---|---|
| `ta_atr_pct` | 339 |
| `regime_score` | 203 |
| `regime_label_encoded` | 185 |
| `mc_vix` | 138 |
| `mc_yield_curve_slope` | 134 |
| `mc_igv_rel_spy_21d` | 108 |
| `vol_rv_60d` | 105 |
| `mc_spy_rv_21d` | 104 |
| `mc_spy_ret_21d` | 104 |
| `fnd_ev_ebitda` | 92 |
| `mc_vix_zscore_60d` | 88 |
| `fnd_eps_growth_q` | 71 |
| `mc_spy_above_200ma` | 68 |
| `fnd_pb` | 66 |
| `fnd_roa` | 62 |
| `ta_sma_200_dist` | 61 |
| `vol_rv_zscore_252d` | 58 |
| `fnd_roe` | 58 |
| `fnd_pct_of_52w_range` | 58 |
| `fnd_ps` | 57 |

## Options overlay summary

- **Trades:** 41
- **Win rate:** 48.8%
- **Total net P&L:** $6,764
- **Avg per trade:** $165

### By strategy

| Strategy | N | Total P&L | Avg P&L | Win rate |
|---|---|---|---|---|
| bull_call_spread | 10 | $5,488 | $549 | 70.0% |
| bull_put_credit | 5 | $-784 | $-157 | 60.0% |
| calendar_spread | 7 | $0 | $0 | 0.0% |
| iron_condor | 6 | $181 | $30 | 33.3% |
| long_call | 2 | $288 | $144 | 50.0% |
| short_strangle | 11 | $1,591 | $145 | 63.6% |

---
_Backtest results are not a guarantee of future performance. Run multiple seeds, multiple sub-periods, and check for regime sensitivity._