"""Performance metrics — v2 fixed for period returns.

v1 BUG: functions assumed daily returns. Portfolio rebalances every 21 days,
so returns are PERIOD returns. v1 produced absurd values (Sharpe 6.17,
annualized return 39,996x). v2 fixes this by accepting `period_days` param.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def annualized_return(returns: pd.Series, period_days: int = 21) -> float:
    """Geometric mean return annualized.

    `returns` is one entry per period (e.g., 21-day rebalance). `period_days`
    tells us how to map period count back to years.
    """
    if returns.empty:
        return 0.0
    cum = (1 + returns).prod()
    n_periods = len(returns)
    years = (n_periods * period_days) / TRADING_DAYS
    if years <= 0:
        return 0.0
    return cum ** (1 / years) - 1


def annualized_volatility(returns: pd.Series, period_days: int = 21) -> float:
    if returns.empty:
        return 0.0
    periods_per_year = TRADING_DAYS / period_days
    return returns.std() * np.sqrt(periods_per_year)


def sharpe_ratio(returns: pd.Series, rf_annual: float = 0.045, period_days: int = 21) -> float:
    if returns.empty or returns.std() == 0:
        return 0.0
    periods_per_year = TRADING_DAYS / period_days
    rf_per_period = rf_annual / periods_per_year
    excess = returns - rf_per_period
    return excess.mean() / returns.std() * np.sqrt(periods_per_year)


def sortino_ratio(returns: pd.Series, rf_annual: float = 0.045, period_days: int = 21) -> float:
    if returns.empty:
        return 0.0
    periods_per_year = TRADING_DAYS / period_days
    rf_per_period = rf_annual / periods_per_year
    excess = returns - rf_per_period
    downside = returns[returns < 0]
    if downside.empty or downside.std() == 0:
        return float('inf')
    return excess.mean() / downside.std() * np.sqrt(periods_per_year)


def max_drawdown(returns: pd.Series) -> tuple[float, int]:
    if returns.empty:
        return 0.0, 0
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    dd = (cum / running_max) - 1
    mdd = dd.min()
    peak_idx = running_max.idxmax()
    trough_idx = dd.idxmin()
    try:
        duration = (trough_idx - peak_idx).days
    except Exception:
        duration = 0
    return float(mdd), int(duration)


def calmar_ratio(returns: pd.Series, period_days: int = 21) -> float:
    ann_ret = annualized_return(returns, period_days)
    mdd, _ = max_drawdown(returns)
    if mdd == 0:
        return float('inf')
    return ann_ret / abs(mdd)


def hit_rate(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    return (returns > 0).mean()


def profit_factor(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    if losses == 0:
        return float('inf')
    return gains / losses


def information_coefficient(predictions: pd.Series, realized: pd.Series) -> float:
    aligned = pd.concat([predictions, realized], axis=1, join='inner').dropna()
    if len(aligned) < 30:
        return 0.0
    return aligned.iloc[:, 0].corr(aligned.iloc[:, 1])


def rank_ic(predictions: pd.Series, realized: pd.Series) -> float:
    aligned = pd.concat([predictions, realized], axis=1, join='inner').dropna()
    if len(aligned) < 30:
        return 0.0
    return aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method='spearman')


def auc_score(predictions: pd.Series, binary_target: pd.Series) -> float:
    """ROC-AUC: how well predictions discriminate up vs down."""
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return float('nan')
    aligned = pd.concat([predictions, binary_target], axis=1, join='inner').dropna()
    if len(aligned) < 30 or aligned.iloc[:, 1].nunique() < 2:
        return float('nan')
    return float(roc_auc_score(aligned.iloc[:, 1].values, aligned.iloc[:, 0].values))


def compute_metrics(
    daily_returns: pd.Series,
    trade_returns: pd.Series | None = None,
    predictions: pd.Series | None = None,
    realized: pd.Series | None = None,
    rf_annual: float = 0.045,
    period_days: int = 21,
) -> dict:
    """Comprehensive metrics dict.

    NOTE: `daily_returns` parameter name is a v1 artifact — in practice the
    series contains PERIOD returns (one per rebalance). `period_days` tells
    us how to annualize correctly.
    """
    mdd, mdd_days = max_drawdown(daily_returns)
    out = {
        'total_return': float((1 + daily_returns).prod() - 1) if not daily_returns.empty else 0.0,
        'annualized_return': annualized_return(daily_returns, period_days),
        'annualized_volatility': annualized_volatility(daily_returns, period_days),
        'sharpe': sharpe_ratio(daily_returns, rf_annual, period_days),
        'sortino': sortino_ratio(daily_returns, rf_annual, period_days),
        'max_drawdown': mdd,
        'max_drawdown_days': mdd_days,
        'calmar': calmar_ratio(daily_returns, period_days),
        'hit_rate_periods': hit_rate(daily_returns),
        'n_periods': len(daily_returns),
        'period_days': period_days,
    }
    if trade_returns is not None and not trade_returns.empty:
        out['n_trades'] = len(trade_returns)
        out['hit_rate_trades'] = hit_rate(trade_returns)
        out['avg_winner'] = float(trade_returns[trade_returns > 0].mean()) if (trade_returns > 0).any() else 0.0
        out['avg_loser'] = float(trade_returns[trade_returns < 0].mean()) if (trade_returns < 0).any() else 0.0
        out['profit_factor'] = profit_factor(trade_returns)
    if predictions is not None and realized is not None:
        out['information_coefficient'] = information_coefficient(predictions, realized)
        out['rank_ic'] = rank_ic(predictions, realized)
        # Binary AUC (predictions vs realized > 0)
        if not realized.empty:
            binary = (realized > 0).astype(int)
            out['auc_up_any'] = auc_score(predictions, binary)
            binary_3pct = (realized > 0.03).astype(int)
            out['auc_up_3pct'] = auc_score(predictions, binary_3pct)
    return out
