"""Options-overlay backtest — v2 compatible with the 7-strategy StrategySelector.

v2 fixes vs v1:
  - self.selector.expiry_dte  -> self.selector.expiry_dte_short  (renamed)
  - direction_score_pct=...   -> direction_prob=...               (renamed)
  - score is now P(up) directly (0-1), not a rank percentile, so we pass
    row['score'] straight through as direction_prob.

Simulates each top-N pick's options trade using simplified Black-Scholes with
realized-vol-derived IV. Applies tier-based options costs. Holds to the next
rebalance.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from math import erf, exp, log, sqrt

import numpy as np
import pandas as pd

from models.strategy_selector import OptionsTrade, StrategySelector
from backtest.costs import OptionsCostModel

logger = logging.getLogger(__name__)


# ---------- Simplified Black-Scholes ----------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + erf(x / sqrt(2)))


def bsm_call(spot: float, strike: float, t_years: float, vol: float, r: float = 0.045) -> float:
    if t_years <= 0 or vol <= 0:
        return max(spot - strike, 0)
    d1 = (log(spot / strike) + (r + 0.5 * vol ** 2) * t_years) / (vol * sqrt(t_years))
    d2 = d1 - vol * sqrt(t_years)
    return spot * _norm_cdf(d1) - strike * exp(-r * t_years) * _norm_cdf(d2)


def bsm_put(spot: float, strike: float, t_years: float, vol: float, r: float = 0.045) -> float:
    call = bsm_call(spot, strike, t_years, vol, r)
    return call - spot + strike * exp(-r * t_years)


def price_leg(leg: dict, spot: float, t_years: float, iv: float) -> float:
    if leg['type'] == 'call':
        return bsm_call(spot, leg['strike'], t_years, iv)
    return bsm_put(spot, leg['strike'], t_years, iv)


def price_strategy(trade: OptionsTrade, spot: float, t_years: float, iv: float) -> float:
    """Net premium (positive = debit, negative = credit)."""
    net = 0.0
    for leg in trade.legs:
        p = price_leg(leg, spot, t_years, iv) * leg['qty']
        if leg['action'] == 'buy':
            net += p
        else:
            net -= p
    return net


@dataclass
class OptionsTradeRecord:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    ticker: str
    strategy: str
    entry_premium: float
    exit_premium: float
    pnl_per_contract: float
    cost: float
    net_pnl: float
    direction_prob: float
    predicted_vol: float
    realized_vol: float


class OptionsBacktester:
    def __init__(
        self,
        strategy_selector: StrategySelector,
        cost_model: OptionsCostModel | None = None,
        iv_to_rv_ratio: float = 1.20,
        early_exit_pct: float = 0.60,
        contract_size: int = 100,
    ):
        self.selector = strategy_selector
        self.cost_model = cost_model or OptionsCostModel()
        self.iv_to_rv_ratio = iv_to_rv_ratio
        self.early_exit_pct = early_exit_pct
        self.contract_size = contract_size

    def run(
        self,
        predictions: pd.DataFrame,
        prices_panel: dict,
        vol_forecasts: pd.DataFrame | None,
        liquidity_tiers: dict,
    ) -> pd.DataFrame:
        predictions = predictions.copy()
        predictions['date'] = pd.to_datetime(predictions['date'])
        all_dates = pd.DatetimeIndex(sorted(predictions['date'].unique()))
        # v2 FIX: expiry_dte -> expiry_dte_short
        step = max(1, self.selector.expiry_dte_short)
        rebalance_dates = all_dates[::step]

        # Map vol forecasts
        vol_lookup = {}
        if vol_forecasts is not None and not vol_forecasts.empty:
            vf = vol_forecasts.copy()
            vf['date'] = pd.to_datetime(vf['date'])
            for _, r in vf.iterrows():
                vol_lookup[(r['date'], r['ticker'])] = r['predicted_vol']

        records: list[OptionsTradeRecord] = []
        for i, entry_date in enumerate(rebalance_dates):
            if i + 1 >= len(rebalance_dates):
                break
            exit_date = rebalance_dates[i + 1]

            day_preds = predictions[predictions['date'] == entry_date]
            day_preds = day_preds.sort_values('score', ascending=False).head(5)

            for _, row in day_preds.iterrows():
                ticker = row['ticker']
                p_up = float(row['score'])  # v2: score IS P(up)
                spot_entry = self._spot(prices_panel, ticker, entry_date)
                spot_exit = self._spot(prices_panel, ticker, exit_date)
                if spot_entry is None or spot_exit is None:
                    continue

                predicted_vol = vol_lookup.get((entry_date, ticker), 0.40)
                realized_vol = self._realized_vol(prices_panel, ticker, entry_date, exit_date)

                # v2 FIX: direction_prob instead of direction_score_pct
                trade = self.selector.select(
                    ticker=ticker,
                    spot=spot_entry,
                    direction_prob=p_up,
                    predicted_vol=predicted_vol,
                    current_rv_21d=realized_vol,
                    iv_rank=None,
                    liquidity_tier=liquidity_tiers.get(ticker, 3),
                )
                if trade.strategy == 'wait' or not trade.legs:
                    continue

                iv_entry = max(predicted_vol * self.iv_to_rv_ratio, 0.10)
                t_entry = max(trade.expiry_dte, 1) / 365
                entry_premium = price_strategy(trade, spot_entry, t_entry, iv_entry)

                t_exit = max((trade.expiry_dte - (exit_date - entry_date).days) / 365, 1 / 365)
                iv_exit = max(realized_vol * self.iv_to_rv_ratio, 0.10)
                exit_premium = price_strategy(trade, spot_exit, t_exit, iv_exit)

                pnl = (exit_premium - entry_premium) * self.contract_size
                cost = self.cost_model.cost_for_tier(
                    liquidity_tiers.get(ticker, 3), n_legs=len(trade.legs),
                ) * abs(entry_premium) * self.contract_size
                net_pnl = pnl - cost

                records.append(OptionsTradeRecord(
                    entry_date=entry_date, exit_date=exit_date, ticker=ticker,
                    strategy=trade.strategy, entry_premium=entry_premium,
                    exit_premium=exit_premium, pnl_per_contract=pnl, cost=cost,
                    net_pnl=net_pnl, direction_prob=p_up,
                    predicted_vol=predicted_vol, realized_vol=realized_vol,
                ))

        return pd.DataFrame([r.__dict__ for r in records])

    @staticmethod
    def _spot(prices_panel, ticker, date):
        df = prices_panel.get(ticker)
        if df is None or df.empty:
            return None
        try:
            return float(df.loc[df.index <= date, 'Close'].iloc[-1])
        except Exception:
            return None

    @staticmethod
    def _realized_vol(prices_panel, ticker, start, end):
        df = prices_panel.get(ticker)
        if df is None or df.empty:
            return 0.30
        try:
            window = df.loc[(df.index >= start - pd.Timedelta(days=30)) & (df.index <= end), 'Close']
            if len(window) < 5:
                return 0.30
            ret = np.log(window).diff().dropna()
            return float(ret.std() * np.sqrt(252))
        except Exception:
            return 0.30
