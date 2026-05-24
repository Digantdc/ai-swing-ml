# Results Summary

All figures are out-of-sample, walk-forward (45 monthly folds, 2022-08 → 2026-05),
net of 15 bps cost + 10 bps slippage + 5 bps live drag. Period-correct annualization.

## 1. Model performance (equity leg, top-5 portfolio)

| Metric | Value | Interpretation |
|---|---|---|
| Total return | 449.4% | Over ~3.7-year test window |
| Annualized return | 59.1% | High — but see benchmark below |
| Annualized volatility | 46.9% | Very high (5-name concentration) |
| **Sharpe** | **1.12** | Decent in isolation; loses to SMH (below) |
| Sortino | 3.70 | Downside-adjusted |
| Max drawdown | −25.8% | Aggressive |
| Calmar | 2.30 | Return / max DD |
| Hit rate (trades) | 54.6% | Slight edge |
| Avg winner / loser | +16.8% / −9.7% | Healthy asymmetry |
| Profit factor | 2.08 | Gross gains / losses |
| **Information Coefficient** | **0.003** | **≈ 0 — no skill** |
| **Rank IC** | **−0.021** | **Negative — no selection edge** |

## 2. The benchmark test (the decisive comparison)

Same window, same costs. Buy-and-hold passive alternatives vs the model:

| Strategy | Annual Return | Sharpe | Max Drawdown | Effort |
|---|---|---|---|---|
| **Buy & hold SMH (semis ETF)** | 48.8% | **1.225** 🥇 | −18.6% | None |
| AI Swing ML model | 59.1% | 1.123 | −25.8% | High |
| Equal-weight 60 AI names | 35.4% | 1.011 | −18.3% | Monthly rebal |
| Buy & hold SPY | 15.8% | 0.779 | −9.7% | None |

**Verdict:** the model's higher raw return is concentration risk, not skill. On a
risk-adjusted basis it is beaten by a passive SMH purchase. The +23.7% "alpha" over
equal-weight collapses to a +0.11 Sharpe difference once risk is accounted for, and
disappears entirely against the sector ETF.

## 3. Feature predictivity (top of the IC analysis)

Spearman IC and binary AUC against the 21-day forward return, ~108 features tested:

| Rank | Feature | Spearman IC | AUC | Family |
|---|---|---|---|---|
| 1 | mc_yield_curve_slope | +0.116 | 0.553 | Macro |
| 2 | mc_vix | +0.081 | 0.546 | Macro |
| 3 | fnd_roe | +0.078 | 0.536 | Fundamental |
| 4 | mc_spy_above_200ma | −0.077 | 0.465 | Macro |
| 5 | fnd_profit_margin | +0.072 | 0.532 | Fundamental |

**The strongest predictors are macro/regime and fundamental quality — NOT the
short-term technical indicators (RSI, MACD, VWAP, Stochastic), which all scored
AUC ≈ 0.50 (random) at the 21-day horizon.**

## 4. Options overlay (simplified BSM P&L, 41 trades)

| Strategy | N | Total P&L | Win rate |
|---|---|---|---|
| bull_call_spread | 10 | +$5,488 | 70.0% |
| short_strangle | 11 | +$1,591 | 63.6% |
| long_call | 2 | +$288 | 50.0% |
| iron_condor | 6 | +$181 | 33.3% |
| calendar_spread | 7 | $0 | — |
| bull_put_credit | 5 | −$784 | 60.0% |

Small sample; rides the same sector beta as the equity leg. Illustrative of the
7-strategy selector, not an independent edge.

## 5. The honest takeaway

Two model architectures, IC-curated features, a corrected metrics pipeline, and a
proper benchmark all converge on the same conclusion: **there is no reliable
stock-selection alpha at the 21-day horizon in this correlated universe.** The one
genuine signal — market-regime timing — is better expressed by simply timing or
holding the sector ETF.

This negative result, rigorously established, is the deliverable.
