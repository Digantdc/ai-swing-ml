"""Top-N equal-weighted portfolio backtest with realistic costs.

Strategy:
    - Each rebalance day, sort tickers by model score (desc)
    - Take top-N picks (equal-weighted)
    - Hold for `rebalance_days` trading days
    - At next rebalance, close all and re-pick

Output: daily equity curve + trade log.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .costs import CostModel

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    ticker: str
    entry_price: float
    exit_price: float
    weight: float
    gross_return: float
    net_return: float
    score: float


class PortfolioBacktester:
    """Top-N equal-weighted portfolio simulator."""

    def __init__(
        self,
        top_n: int = 5,
        rebalance_days: int = 21,
        cost_model: CostModel | None = None,
        weighting: str = 'equal',  # 'equal' or 'rank'
        min_score_pct: float = 0.10,
    ):
        self.top_n = top_n
        self.rebalance_days = rebalance_days
        self.cost_model = cost_model or CostModel()
        self.weighting = weighting
        self.min_score_pct = min_score_pct

    def run(
        self,
        predictions: pd.DataFrame,
        prices_panel: dict[str, pd.DataFrame],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run the backtest.

        Args:
            predictions: DataFrame with columns ['date', 'ticker', 'score']
                Predictions for every (date, ticker) combo.
            prices_panel: dict {ticker: OHLCV df with 'Close' column}

        Returns:
            (daily_equity_df, trades_df)
            daily_equity_df: ['date', 'equity', 'daily_return']
            trades_df: list of trades
        """
        predictions = predictions.copy()
        predictions['date'] = pd.to_datetime(predictions['date'])

        # Build long price dataframe (date, ticker, close)
        price_rows = []
        for t, df in prices_panel.items():
            tmp = df[['Close']].copy()
            tmp['ticker'] = t
            tmp = tmp.reset_index().rename(columns={'index': 'date', 'Date': 'date'})
            tmp.columns = [c if c != 'Close' else 'close' for c in tmp.columns]
            price_rows.append(tmp[['date', 'ticker', 'close']])
        prices = pd.concat(price_rows, ignore_index=True)
        prices['date'] = pd.to_datetime(prices['date'])

        # Determine rebalance dates (every `rebalance_days` trading days within prediction range)
        all_dates = pd.DatetimeIndex(sorted(predictions['date'].unique()))
        if all_dates.empty:
            return pd.DataFrame(), pd.DataFrame()
        rebalance_dates = all_dates[::self.rebalance_days]

        trades: list[Trade] = []
        for i, rb_date in enumerate(rebalance_dates):
            # Skip last rebalance if no future data
            if i + 1 >= len(rebalance_dates):
                break
            exit_date = rebalance_dates[i + 1]

            # Get top-N picks at rebalance date
            day_preds = predictions[predictions['date'] == rb_date].copy()
            if day_preds.empty:
                continue
            day_preds = day_preds.sort_values('score', ascending=False)
            # Filter by min score percentile
            score_threshold = day_preds['score'].quantile(1 - self.min_score_pct)
            day_preds = day_preds[day_preds['score'] >= score_threshold]
            picks = day_preds.head(self.top_n)
            if picks.empty:
                continue

            weight = 1.0 / len(picks) if self.weighting == 'equal' else None

            for _, row in picks.iterrows():
                ticker = row['ticker']
                entry_p = self._get_price(prices, ticker, rb_date)
                exit_p = self._get_price(prices, ticker, exit_date)
                if entry_p is None or exit_p is None or entry_p <= 0:
                    continue
                gross = (exit_p / entry_p) - 1
                net = self.cost_model.apply_to_return(gross)
                trades.append(Trade(
                    entry_date=rb_date,
                    exit_date=exit_date,
                    ticker=ticker,
                    entry_price=entry_p,
                    exit_price=exit_p,
                    weight=weight,
                    gross_return=gross,
                    net_return=net,
                    score=row['score'],
                ))

        # Build daily equity curve
        if not trades:
            return pd.DataFrame(columns=['date', 'equity', 'daily_return']), pd.DataFrame()

        equity_curve = self._build_equity_curve(trades, all_dates)
        trades_df = pd.DataFrame([t.__dict__ for t in trades])
        return equity_curve, trades_df

    @staticmethod
    def _get_price(prices: pd.DataFrame, ticker: str, date: pd.Timestamp) -> float | None:
        match = prices[(prices['ticker'] == ticker) & (prices['date'] == date)]
        if match.empty:
            # Find next available date within 3 days
            forward = prices[(prices['ticker'] == ticker) & (prices['date'] > date)].head(3)
            if not forward.empty:
                return float(forward.iloc[0]['close'])
            return None
        return float(match.iloc[0]['close'])

    def _build_equity_curve(self, trades: list[Trade], all_dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Daily-mark-to-market equity curve from trade list."""
        # Bucket trades by entry date for simplified daily P&L attribution
        df = pd.DataFrame([t.__dict__ for t in trades])
        # Each trade contributes weight * net_return at exit_date
        # For simplicity, mark-to-market at exit only (period return aggregation)
        pnl_by_exit = df.groupby('exit_date').apply(
            lambda x: (x['weight'] * x['net_return']).sum()
        )
        equity = (1 + pnl_by_exit).cumprod()
        out = pd.DataFrame({
            'date': equity.index,
            'equity': equity.values,
            'period_return': pnl_by_exit.values,
        })
        out['daily_return'] = out['period_return']  # period == rebalance window
        return out
