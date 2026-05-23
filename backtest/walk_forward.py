"""Walk-forward time-series cross-validation.

Generates a sequence of (train_idx, valid_idx, test_idx) tuples that step
forward through time. No look-ahead. Suitable for monthly retraining.

Example with total_years=5, initial_train_months=24, validation_months=3,
retrain_freq_months=1:

    fold 0: train [t-27m, t-3m], valid [t-3m, t], test [t, t+1m]
    fold 1: train [t-26m+1m, t-2m+1m], valid [t-2m+1m, t+1m], test [t+1m, t+2m]
    ...
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


class WalkForwardBacktest:
    """Generate walk-forward folds for time-series CV."""

    def __init__(
        self,
        initial_train_months: int = 24,
        validation_months: int = 3,
        retrain_freq_months: int = 1,
        test_horizon_months: int = 1,
    ):
        self.initial_train_months = initial_train_months
        self.validation_months = validation_months
        self.retrain_freq_months = retrain_freq_months
        self.test_horizon_months = test_horizon_months

    def generate_folds(
        self,
        dates: pd.DatetimeIndex,
    ) -> list[Fold]:
        """Generate fold boundaries given a sorted DatetimeIndex of all data dates."""
        dates = pd.DatetimeIndex(sorted(set(dates)))
        if dates.empty:
            return []
        start = dates.min()
        end = dates.max()

        # First test starts after initial train + validation
        first_test_start = start + pd.DateOffset(months=self.initial_train_months + self.validation_months)
        if first_test_start >= end:
            return []

        folds = []
        cur_test_start = first_test_start
        while cur_test_start < end:
            test_end = min(cur_test_start + pd.DateOffset(months=self.test_horizon_months), end)
            valid_end = cur_test_start
            valid_start = valid_end - pd.DateOffset(months=self.validation_months)
            train_end = valid_start
            train_start = start  # expanding window; switch to rolling if desired
            folds.append(Fold(
                train_start=train_start,
                train_end=train_end,
                valid_start=valid_start,
                valid_end=valid_end,
                test_start=cur_test_start,
                test_end=test_end,
            ))
            cur_test_start = cur_test_start + pd.DateOffset(months=self.retrain_freq_months)
        return folds

    def split_panel(
        self,
        panel: pd.DataFrame,
        fold: Fold,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Slice a panel DataFrame (must have 'date' column) into train/valid/test."""
        m = panel['date']
        train = panel[(m >= fold.train_start) & (m < fold.train_end)]
        valid = panel[(m >= fold.valid_start) & (m < fold.valid_end)]
        test = panel[(m >= fold.test_start) & (m < fold.test_end)]
        return train, valid, test
