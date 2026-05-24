# AI Swing ML — A Quant Research Study in Honest Validation

**Can a machine-learning model systematically pick 21-day winners from a basket of
~60 AI/semiconductor stocks?** I built a full research pipeline to answer that — and
the disciplined answer is **no**. This repository documents the build, the rigorous
validation that exposed the truth, and the lessons that matter more than a polished
positive would.

> ### The result, stated honestly
> The final model posted a backtest **Sharpe of 1.12** — a number that *looks* like
> success. But a one-click **passive SMH (semiconductor ETF) buy-and-hold beat it:
> Sharpe 1.23 with a shallower drawdown (−18.6% vs −25.8%).** The model's
> stock-selection skill, measured by **Rank IC, was −0.021 (statistically zero).** Its
> apparent returns were AI-sector beta, not alpha.
>
> **I consider finding and proving this the success of the project.** Most retail quant
> projects never run the benchmark and end up trading an over-fit model into a loss.
> This repo is a demonstration of *research process and intellectual honesty* — the
> things that actually matter on a quant desk.

---

## What this project demonstrates

Built end-to-end, this project shows I can:

- **Build a complete, modular quant pipeline** from data to signal to backtest to
  portfolio — not a notebook, a structured codebase with tests.
- **Validate the right way.** Walk-forward CV with no look-ahead, realistic transaction
  costs, and — critically — a benchmark comparison that most candidates skip.
- **Measure skill, not just returns.** I report Information Coefficient and Rank IC, not
  just Sharpe, because I know returns in a bull market lie.
- **Find my own mistakes.** I caught a metrics-annualization bug that had inflated Sharpe
  to a fake 6.17, and I diagnosed look-ahead bias in the fundamental data feed.
- **Reason about *why* a model fails** — universe correlation, horizon selection,
  survivorship and point-in-time data — and propose concrete fixes.

If you want to see how I think, read the **[Research Narrative](#the-research-narrative)**
and **[Why It Failed](#why-it-failed-root-cause-analysis)** sections below.

---

## The research narrative

**Hypothesis.** AI/semiconductor names trend hard (e.g., GlobalFoundries ran $40→$71 in
weeks). If a model could rank which names will outperform over the next 3–4 weeks, a
swing trader could capture those moves systematically.

**v1 — Cross-sectional ranker (LightGBM LambdaRank).** Trained to rank the 60 names by
21-day forward return. Result: **Rank IC −0.017** — worse than random. Diagnosis: the
names are too correlated (0.6–0.8 pairwise); there is little to rank between them.

**v2 — Per-stock binary classifier (XGBoost).** Reframed the problem as
"P(this stock is up >3% in 21 days?)", trained on a wider 150-name, 7-sector universe to
break the sector-only bias, and curated features by Information Coefficient (dropping
RSI/MACD/VWAP, which scored AUC ≈ 0.50 — pure noise at this horizon). Result: a
believable **Sharpe of 1.12** and positive trade stats — but **Rank IC still −0.021.**

**The benchmark test (the decider).** I compared the model to passive alternatives over
the identical window with identical costs. SMH buy-and-hold won on a risk-adjusted basis.
The model's Sharpe was sector beta, amplified by 5-name concentration — not selection
skill. **The model is not worth trading over the ETF.**

**The honest conclusion.** Systematic 21-day stock selection in a tightly-correlated AI
universe, using freely-available data, is at or beyond the limit of predictability. The
only genuine signal the model found was *market-regime timing* (VIX, yield curve, trend),
not *stock selection*.

---

## Key technical findings

| Finding | Evidence |
|---|---|
| No stock-selection edge at the 21-day horizon | Rank IC −0.021, IC 0.003 (both ≈ 0), across two architectures |
| Apparent Sharpe is sector beta, not alpha | Model Sharpe 1.12 < SMH buy-and-hold 1.23; model drawdown deeper (−25.8% vs −18.6%) |
| Short-term technicals are noise at 21d | RSI/MACD/VWAP all AUC ≈ 0.50 in the IC analysis |
| Predictive signal is macro/regime, not stock-specific | Feature importance dominated by VIX, yield-curve slope, regime score |
| Self-caught methodology bugs | Fixed a period-vs-day annualization bug (false Sharpe 6.17 → real 1.12) |

Full numbers: see **[results/RESULTS.md](results/RESULTS.md)**.

---

## Why it failed (root-cause analysis)

1. **Universe correlation.** 0.6–0.8 pairwise correlation among the 60 names leaves almost
   no idiosyncratic return to predict — cross-sectional selection has nothing to grip.
2. **Horizon dead zone.** 21 days is too long for microstructure/momentum signals and too
   short for fundamental factors to express in price.
3. **Look-ahead bias in fundamentals.** `yfinance` `.info` returns a *current* snapshot;
   broadcasting it across history leaks future knowledge. Notably, even *with* this
   advantage the IC was ~0 — the true (point-in-time) result would be worse.
4. **Survivorship + single regime.** The 60 names are today's winners, over a 2022–2026
   window dominated by one AI bull market. The model never saw a tech bear.

---

## How a rigorous v3 would fix it

1. **Point-in-time fundamentals** (CRSP/Compustat) — eliminate the look-ahead leak
2. **Survivorship-free universe** including delisted names
3. **Diversified, lower-correlation universe** so selection has real dispersion
4. **Longer horizon (60d)** or **residualized targets** (excess return vs sector ETF) to
   strip beta and isolate alpha
5. **A regime-timed ETF strategy** — operationalize the one signal that was real

---

## What's inside

- **Walk-forward time-series CV** — 24m train / 3m validation / monthly retrain → 45
  out-of-sample folds, no look-ahead in the split
- **~108 engineered features** across 6 families (price, technical, volume, volatility,
  fundamental, macro/regime, calendar), curated to ~33 by Information Coefficient
- **Two model architectures** — LightGBM LambdaRank vs XGBoost binary classifier with
  isotonic probability calibration
- **Realistic backtester** — 15 bps cost + 10 bps slippage + live drag, period-correct
  annualization, drawdown/Sharpe/Sortino/Calmar, IC/Rank IC/AUC
- **Options-strategy engine** — rule-based mapping of (direction, vol, IV regime, earnings
  proximity) → 7 strategies (bull call/put spreads, PMCC, short strangle, iron condor,
  earnings iron condor, calendar) with simplified Black-Scholes P&L
- **Risk modules** — rule-based regime detector + Kelly position sizer with caps
- **50+ unit tests** covering features, metrics, strategy routing, and the metrics fix

## Tech stack

`Python` · `pandas` · `numpy` · `scikit-learn` · `XGBoost` · `LightGBM` · `scipy` ·
`yfinance` · `pyarrow` — gradient-boosted trees, walk-forward CV, isotonic calibration

## Project structure

```
ai_swing_ml/
├── data/               # universe definitions + cached yfinance fetcher
├── features/           # 6 feature families + IC-curated feature set
├── models/             # XGBoost classifier, RF vol model, regime detector,
│                       #   Kelly sizer, 7-strategy options selector
├── backtest/           # walk-forward CV, portfolio sim, options overlay,
│                       #   cost model, metrics (IC, Sharpe, AUC, drawdown)
├── reports/ results/   # report generation + committed example outputs
├── tests/              # 50+ unit tests
├── train.py            # fit production model
├── predict.py          # daily inference → picks + options recommendations
├── run_backtest.py     # full walk-forward backtest
├── analyze_features.py # feature predictivity analysis (IC / AUC / MI)
└── benchmark.py        # the decisive test: model vs passive buy-and-hold
```

## Reproduce it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python tests/test_v2.py            # unit tests
python run_backtest.py --years 5   # full backtest  → output/backtest_report.md
python benchmark.py --years 5      # model vs SMH / SPY / equal-weight
python analyze_features.py         # feature IC / AUC ranking
```

## Disclaimer

Research and educational project. Not investment advice. Backtested performance does not
predict future results, and (as this study demonstrates) can be misleading without proper
benchmark validation.

---

*Built by Digant Chaudhari. The most valuable line in this repo is the one that says the
model didn't beat the ETF — because it's true, and I can prove it.*
