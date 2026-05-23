"""Realized volatility forecasting (RF regressor).

Target: forward 21-day realized vol (annualized).
Used to size options trades and pick the right strategy
(low predicted vol + low IV rank = long premium; high predicted vol = sell premium).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Re-use feature column logic
from .ranker import META_COLS, get_feature_columns


class VolatilityModel:
    """RF regressor for forward 21d realized volatility."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 8,
        min_samples_leaf: int = 20,
        random_state: int = 42,
    ):
        self.rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
        )
        self.feature_cols: list[str] = []

    def fit(self, train: pd.DataFrame) -> 'VolatilityModel':
        train = train.dropna(subset=['target_fwd_vol']).copy()
        self.feature_cols = get_feature_columns(train)

        # Drop categorical for RF (or handle separately)
        feature_cols_use = [c for c in self.feature_cols if c != 'sector']

        X = train[feature_cols_use].fillna(train[feature_cols_use].median(numeric_only=True))
        # If any column is still all NaN, drop it
        keep = [c for c in feature_cols_use if X[c].notna().any()]
        X = X[keep]
        self.feature_cols = keep

        y = train['target_fwd_vol']
        self.rf.fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        X = df[self.feature_cols].fillna(
            df[self.feature_cols].median(numeric_only=True)
        )
        preds = self.rf.predict(X)
        return pd.Series(preds, index=df.index, name='predicted_vol')

    def feature_importance(self) -> pd.DataFrame:
        return pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.rf.feature_importances_,
        }).sort_values('importance', ascending=False).reset_index(drop=True)

    def save(self, path: str | Path):
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({'rf': self.rf, 'feature_cols': self.feature_cols}, path)

    @classmethod
    def load(cls, path: str | Path) -> 'VolatilityModel':
        import joblib
        bundle = joblib.load(path)
        obj = cls()
        obj.rf = bundle['rf']
        obj.feature_cols = bundle['feature_cols']
        return obj
