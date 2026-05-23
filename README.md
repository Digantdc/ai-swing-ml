# AI Swing ML — Cross-Sectional Equity & Options Research Pipeline

A production-structured machine-learning research pipeline for 21-day swing trading
in the AI / semiconductor / autonomous-systems equity universe, with an integrated
options-strategy layer. Built end-to-end: data ingestion, feature engineering,
walk-forward model training, realistic backtesting, and — critically — rigorous
benchmark validation.

> **Headline result (and why it matters):** After two model architectures, IC-based
> feature selection, and a fixed-cost walk-forward backtest, the model achieved a
> Sharpe of ~1.12 — but **a passive SMH (semiconductor ETF) buy-and-hold beat it on a
> risk-adjusted basis (Sharpe 1.23, smaller drawdown).** The project's value is in
> *proving* this with proper methodology rather than reporting an over-fit positive.
> This repo demonstrates quant research process, not a get-rich strategy.

---

## What this project demonstrates

- **End-to-end quant research pipeline** — data → features → model → backtest → benchmark
- **Walk-forward time-series cross-validation** (24-month train / 3-month validation /
  monthly retrain → 45 out-of-sample folds; no look-ahead in the CV split)
- **Cross-sectional feature analysis** — Spearman IC, Pearson, mutual information, and
  binary AUC computed for ~108 engineered features to select the predictive ~33
- **Two model architectures compared** — LightGBM LambdaRank (cross-sectional ranker)
  vs XGBoost binary classifier with isotonic probability calibration
- **Realistic backtesting** — transaction costs (15 bps) + slippage (10 bps) + live drag,
  period-correct annualization, and a benchmark comparison against equal-weight and ETF
- **Options-strategy engine** — rule-based mapping of (direction, volatility, IV regime,
  earnings proximity) to 7 strategies (bull call/put spreads, PMCC, short strangle,
  iron condor, earnings iron condor, calendar spread) with simplified Black-Scholes P&L
- **Risk management modules** — rule-based regime detector, Kelly position sizer with
  per-position and gross-exposure caps
- **Honest validation** — the model is benchmarked against passive alternatives and the
  negative result is reported transparently (most retail quant projects skip this step)

## Key technical findings

| Finding | Evidence |
|---|---|
| No stock-selection edge at 21-day horizon | Rank IC −0.021, IC 0.003 (both ≈ 0) |
| Apparent Sharpe is AI-sector beta, not alpha | Model Sharpe 1.12 < SMH buy-and-hold 1.23 |
| Short-term technicals (RSI/MACD/VWAP) are noise at 21d | AUC ≈ 0.50 for all in IC analysis |
| Strongest signals are macro/regime, not stock-specific | Feature importance dominated by VIX, yield curve, regime score |
| Diagnosed a metrics annualization bug | Caught a false Sharpe 6.17; corrected to 1.12 |

These align with the academic literature: single-stock direction at multi-week horizons
in a highly-correlated sector is near the limit of predictability.

## Tech stack

`Python` · `pandas` · `numpy` · `scikit-learn` · `XGBoost` · `LightGBM` · `scipy` ·
`yfinance` · `pyarrow` · walk-forward CV · gradient-boosted trees · isotonic calibration

## Project structure

```
ai_swing_ml/
├── data/              # universe definitions + yfinance fetcher (cached)
├── features/          # 6 feature families + IC-curated feature set
├── models/            # XGBoost classifier, RF vol model, regime detector,
│                      #   Kelly sizer, 7-strategy options selector
├── backtest/          # walk-forward CV, portfolio sim, options overlay,
│                      #   cost model, metrics (IC, Sharpe, AUC, drawdown)
├── reports/           # markdown report generation
├── tests/             # 50+ unit tests (features, backtest, strategies, metrics)
├── train.py           # fit production model
├── predict.py         # daily inference → top picks + options recommendations
├── run_backtest.py    # full walk-forward backtest
├── analyze_features.py # feature predictivity analysis (IC/AUC/MI)
└── benchmark.py       # the decisive test: model vs passive buy-and-hold
```

## Setup & run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python tests/test_v2.py          # unit tests
python run_backtest.py --years 5 # full backtest → output/backtest_report.md
python benchmark.py --years 5    # model vs SMH/SPY/equal-weight
python analyze_features.py       # feature IC/AUC ranking
```

## What I'd do differently (next iteration)

A rigorous v3 would address the flaws this project surfaced:
1. **Point-in-time fundamentals** (CRSP/Compustat) to eliminate the look-ahead bias of
   broadcasting current yfinance fundamentals across history
2. **Survivorship-free universe** including delisted names
3. **Diversified, lower-correlation universe** so cross-sectional selection has dispersion
4. **Longer horizon (60d)** where fundamental factors express, or **residualized targets**
   (excess return vs sector ETF) to strip out beta
5. **A regime-timed ETF strategy** — the one signal the model genuinely found

## Disclaimer

Research and educational project. Not investment advice. Past backtested performance
does not predict future results.
