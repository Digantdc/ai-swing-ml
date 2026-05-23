"""Master feature builder — orchestrates all feature modules.

Takes raw OHLCV panel + fundamentals + macro data and produces a long-format
feature DataFrame ready for cross-sectional ML:

    columns: ticker, date, [80+ feature columns], target_*
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .calendar import compute_calendar_features
from .fundamental import compute_fundamental_features
from .macro import compute_macro_features, compute_relative_strength
from .price import compute_price_features
from .technical import compute_technical_features
from .volatility import compute_forward_realized_vol, compute_volatility_features
from .volume import compute_volume_features

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """Orchestrate feature computation across the universe."""

    def __init__(
        self,
        return_windows: tuple[int, ...] = (1, 5, 10, 20, 60, 120),
        vol_windows: tuple[int, ...] = (5, 21, 60),
        target_horizon: int = 21,
        target_top_pct: float = 0.20,
        sector_map: Optional[dict[str, str]] = None,
    ):
        self.return_windows = return_windows
        self.vol_windows = vol_windows
        self.target_horizon = target_horizon
        self.target_top_pct = target_top_pct
        self.sector_map = sector_map or {}

    # ---------------------------------------------------- per-ticker pipeline

    def build_single_ticker(
        self,
        ticker: str,
        ohlcv: pd.DataFrame,
        fundamentals: dict,
        macro_features: pd.DataFrame,
        benchmark_close: pd.Series,
        next_earnings_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Compute all features for a single ticker.

        Returns long-format DataFrame indexed by date, includes 'ticker' column.
        """
        if ohlcv.empty or len(ohlcv) < 200:
            logger.debug(f"Skipping {ticker}: insufficient data ({len(ohlcv)} rows)")
            return pd.DataFrame()

        frames = []
        frames.append(compute_price_features(ohlcv, list(self.return_windows)))
        frames.append(compute_technical_features(ohlcv))
        frames.append(compute_volume_features(ohlcv))
        frames.append(compute_volatility_features(ohlcv, tuple(self.vol_windows)))

        # Fundamentals — static, broadcast as columns
        fnd = compute_fundamental_features(
            fundamentals,
            current_price=ohlcv['Close'].iloc[-1],
        )
        fnd_df = pd.DataFrame(
            {k: [v] * len(ohlcv) for k, v in fnd.items()},
            index=ohlcv.index,
        )
        frames.append(fnd_df)

        # Macro (broadcast common macro panel to this ticker's index)
        if not macro_features.empty:
            macro_aligned = macro_features.reindex(ohlcv.index).ffill()
            frames.append(macro_aligned)

        # Relative strength vs benchmark
        if benchmark_close is not None and not benchmark_close.empty:
            rs = compute_relative_strength(ohlcv['Close'], benchmark_close)
            frames.append(rs)

        # Calendar
        cal_feats = compute_calendar_features(ohlcv.index, next_earnings_date)
        frames.append(cal_feats)

        # Combine
        combined = pd.concat(frames, axis=1)
        combined['ticker'] = ticker
        combined['close'] = ohlcv['Close']
        # Add sector as categorical feature
        if ticker in self.sector_map:
            combined['sector'] = self.sector_map[ticker]
        else:
            combined['sector'] = 'unknown'

        return combined

    # ------------------------------------------------- universe-level pipeline

    def build_panel(
        self,
        ohlcv_panel: dict[str, pd.DataFrame],
        fundamentals_panel: dict[str, dict],
        macro_panel: dict[str, pd.DataFrame],
        earnings_dates: dict[str, pd.Timestamp] | None = None,
        benchmark: str = 'SPY',
    ) -> pd.DataFrame:
        """Compute features across the universe, return long-format DataFrame.

        Output columns include: ticker, date (index), close, ~80 features.
        """
        earnings_dates = earnings_dates or {}
        macro_features = compute_macro_features(macro_panel)
        benchmark_close = (
            macro_panel.get(benchmark, pd.DataFrame()).get('Close', pd.Series())
        )

        all_features = []
        for ticker, df in ohlcv_panel.items():
            feats = self.build_single_ticker(
                ticker,
                df,
                fundamentals_panel.get(ticker, {}),
                macro_features,
                benchmark_close,
                earnings_dates.get(ticker),
            )
            if not feats.empty:
                all_features.append(feats)

        if not all_features:
            return pd.DataFrame()

        panel = pd.concat(all_features, axis=0)
        panel.index.name = 'date'
        panel = panel.reset_index()
        return panel

    # ------------------------------------------------------ target generation

    def add_targets(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Compute cross-sectional rank target and per-stock forward return.

        Adds:
            target_fwd_return: 21d forward log return
            target_fwd_vol: 21d forward realized vol
            target_rank: rank (1..N) of fwd return on the cross-section that date
            target_top_pct: 1 if in top X% by fwd return, 0 otherwise
        """
        panel = panel.copy()
        panel = panel.sort_values(['ticker', 'date'])
        panel['target_fwd_return'] = (
            panel.groupby('ticker')['close']
            .transform(lambda s: np.log(s).shift(-self.target_horizon) - np.log(s))
        )

        # Forward realized vol (target for vol model)
        panel['target_fwd_vol'] = (
            panel.groupby('ticker')['close']
            .transform(
                lambda s: (
                    np.log(s).diff().shift(-self.target_horizon)
                    .rolling(self.target_horizon).std() * np.sqrt(252)
                )
            )
        )

        # Cross-sectional rank target
        panel['target_rank_pct'] = (
            panel.groupby('date')['target_fwd_return']
            .rank(pct=True, ascending=True)
        )
        panel['target_top_pct'] = (
            panel['target_rank_pct'] >= (1 - self.target_top_pct)
        ).astype(int)
        # LightGBM LambdaRank needs integer relevance scores: bucket into 5 tiers
        panel['target_relevance'] = pd.qcut(
            panel['target_rank_pct'].fillna(0.5),
            q=5,
            labels=[0, 1, 2, 3, 4],
            duplicates='drop',
        ).astype(int)

        return panel
