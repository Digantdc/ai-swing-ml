#!/usr/bin/env python3
"""benchmark.py — the make-or-break test: does the model beat buy-and-hold?

Computes three benchmarks over the SAME period and cost model as the backtest:
  1. Equal-weight buy-and-hold of all 60 AI trading-universe names (monthly rebal)
  2. Buy-and-hold SPY (broad market)
  3. Buy-and-hold SMH (semiconductor ETF — closest sector proxy)

Then compares to the model's reported equity-leg performance.

The decisive number is ALPHA = model_return - equal_weight_return.
If alpha is near zero, the model's Sharpe is just AI-sector beta and the ML
adds no value over holding the basket.

Usage:
    python benchmark.py --config config.yaml --years 5
    # Then paste the comparison table back to Claude.
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yaml

from data.fetcher import DataFetcher
from data.training_universe import get_trading_tickers
from backtest.costs import CostModel
from backtest.metrics import (
    annualized_return, annualized_volatility, sharpe_ratio,
    sortino_ratio, max_drawdown, calmar_ratio,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger('benchmark')


def period_returns_equal_weight(
    ohlcv: dict[str, pd.DataFrame],
    rebalance_days: int,
    cost: CostModel,
) -> pd.Series:
    """Equal-weight, periodically-rebalanced buy-and-hold of a basket."""
    # Build aligned close-price matrix
    closes = {}
    for t, df in ohlcv.items():
        if df is None or df.empty:
            continue
        closes[t] = df['Close']
    px = pd.DataFrame(closes).sort_index().ffill()
    if px.empty:
        return pd.Series(dtype=float)

    dates = px.index
    rebal_idx = list(range(0, len(dates), rebalance_days))
    period_rets = []
    period_dates = []
    for i in range(len(rebal_idx) - 1):
        start_i = rebal_idx[i]
        end_i = rebal_idx[i + 1]
        start_px = px.iloc[start_i]
        end_px = px.iloc[end_i]
        # Per-stock period return, equal-weighted across names with valid data
        valid = start_px.notna() & end_px.notna() & (start_px > 0)
        if valid.sum() == 0:
            continue
        stock_rets = (end_px[valid] / start_px[valid]) - 1
        port_ret = stock_rets.mean()
        # Apply round-trip cost (rebalancing churns the whole book)
        port_ret = cost.apply_to_return(port_ret)
        period_rets.append(port_ret)
        period_dates.append(dates[end_i])
    return pd.Series(period_rets, index=pd.DatetimeIndex(period_dates))


def period_returns_single(close: pd.Series, rebalance_days: int, cost: CostModel) -> pd.Series:
    """Buy-and-hold a single asset, marked at each rebalance (for fair comparison)."""
    close = close.sort_index().ffill()
    dates = close.index
    rebal_idx = list(range(0, len(dates), rebalance_days))
    rets, rdates = [], []
    for i in range(len(rebal_idx) - 1):
        s, e = rebal_idx[i], rebal_idx[i + 1]
        if close.iloc[s] > 0:
            r = (close.iloc[e] / close.iloc[s]) - 1
            # Single buy-and-hold has minimal churn; apply cost once at entry only
            rets.append(r)
            rdates.append(dates[e])
    return pd.Series(rets, index=pd.DatetimeIndex(rdates))


def summarize(returns: pd.Series, label: str, period_days: int, rf: float) -> dict:
    return {
        'strategy': label,
        'total_return': float((1 + returns).prod() - 1) if not returns.empty else 0.0,
        'annualized_return': annualized_return(returns, period_days),
        'annualized_vol': annualized_volatility(returns, period_days),
        'sharpe': sharpe_ratio(returns, rf, period_days),
        'sortino': sortino_ratio(returns, rf, period_days),
        'max_drawdown': max_drawdown(returns)[0],
        'calmar': calmar_ratio(returns, period_days),
        'n_periods': len(returns),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config.yaml')
    p.add_argument('--years', type=int, default=5)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    fetcher = DataFetcher(cfg['output']['data_cache_dir'])
    rebalance_days = cfg['backtest']['rebalance_days']
    rf = cfg['backtest']['cash_return_annual']
    cost = CostModel(
        cost_per_leg=cfg['backtest']['cost_per_leg'],
        slippage=cfg['backtest']['slippage'],
        live_drag=cfg['realism']['live_drag'],
    )

    # Match the backtest's TEST window: skip the initial train+valid period.
    # Backtest used initial_train_months + validation_months before first test.
    skip_months = cfg['backtest']['initial_train_months'] + cfg['backtest']['validation_months']
    end_date = datetime.now()
    full_start = end_date - timedelta(days=int(365 * (args.years + 1)))
    test_start = full_start + pd.DateOffset(months=skip_months)

    trading = get_trading_tickers()
    logger.info(f"Fetching {len(trading)} trading-universe names for benchmark...")
    ohlcv = fetcher.get_ohlcv_batch(trading, full_start, end_date, progress=False)
    # Trim each series to the test window for apples-to-apples comparison
    ohlcv_test = {}
    for t, df in ohlcv.items():
        if df is not None and not df.empty:
            ohlcv_test[t] = df[df.index >= test_start]

    spy = fetcher.get_ohlcv('SPY', full_start, end_date)
    smh = fetcher.get_ohlcv('SMH', full_start, end_date)
    spy_test = spy[spy.index >= test_start]['Close'] if not spy.empty else pd.Series(dtype=float)
    smh_test = smh[smh.index >= test_start]['Close'] if not smh.empty else pd.Series(dtype=float)

    results = []

    eq_rets = period_returns_equal_weight(ohlcv_test, rebalance_days, cost)
    results.append(summarize(eq_rets, 'Equal-weight 60 AI names', rebalance_days, rf))

    if not spy_test.empty:
        spy_rets = period_returns_single(spy_test, rebalance_days, cost)
        results.append(summarize(spy_rets, 'Buy & hold SPY', rebalance_days, rf))
    if not smh_test.empty:
        smh_rets = period_returns_single(smh_test, rebalance_days, cost)
        results.append(summarize(smh_rets, 'Buy & hold SMH (semis)', rebalance_days, rf))

    df = pd.DataFrame(results)

    print("\n" + "=" * 78)
    print("  BENCHMARK COMPARISON  (test window only, same costs as backtest)")
    print("=" * 78)
    pd.set_option('display.float_format', lambda x: f'{x:.3f}')
    print(df.to_string(index=False))

    print("\n" + "-" * 78)
    print("  COMPARE TO YOUR MODEL: annualized 59.14%, Sharpe 1.123, maxDD -25.76%")
    print("-" * 78)
    eq = results[0]
    model_ann, model_sharpe = 0.5914, 1.123
    alpha = model_ann - eq['annualized_return']
    sharpe_diff = model_sharpe - eq['sharpe']
    print(f"  Model annualized:        {model_ann:.2%}")
    print(f"  Equal-weight annualized: {eq['annualized_return']:.2%}")
    print(f"  ALPHA (model - eqwt):    {alpha:+.2%}")
    print(f"  Model Sharpe:            {model_sharpe:.3f}")
    print(f"  Equal-weight Sharpe:     {eq['sharpe']:.3f}")
    print(f"  Sharpe improvement:      {sharpe_diff:+.3f}")
    print()
    if alpha > 0.10 and sharpe_diff > 0.2:
        print("  VERDICT: Model meaningfully beats buy-and-hold. The regime timing")
        print("           adds real value. Worth paper-trading.")
    elif alpha > 0.0 and sharpe_diff > 0.0:
        print("  VERDICT: Model modestly beats buy-and-hold. Marginal value — the")
        print("           edge is small relative to complexity.")
    else:
        print("  VERDICT: Model does NOT beat buy-and-hold. The Sharpe is pure AI-sector")
        print("           beta. Just hold an equal-weight basket or SMH/SOXX ETF.")
    print("=" * 78)

    # Save
    df.to_csv('output/benchmark_comparison.csv', index=False)
    logger.info("Saved output/benchmark_comparison.csv")


if __name__ == '__main__':
    main()
