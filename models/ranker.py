"""LightGBM LambdaRank — cross-sectional ranking model.

Target: relevance (0-4 bucket) of 21-day forward return percentile.
Groups: one group per date (each date is a learning-to-rank query).

For inference, we output a continuous score per (date, ticker) that ranks
how likely the stock is to be in the top fwd-return percentile.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Columns that are metadata, not features
META_COLS = {
    'ticker', 'date', 'close', 'sector',
    'target_fwd_return', 'target_fwd_vol',
    'target_rank_pct', 'target_top_pct', 'target_relevance',
}


def get_feature_columns(panel: pd.DataFrame) -> list[str]:
    """Return list of feature column names (excludes metadata + targets)."""
    return [c for c in panel.columns if c not in META_COLS]


class LightGBMRanker:
    """Wrapper around lightgbm.LGBMRanker with cross-sectional ranking setup."""

    def __init__(
        self,
        params: dict | None = None,
        num_boost_round: int = 500,
        early_stopping_rounds: int = 50,
        random_state: int = 42,
    ):
        try:
            import lightgbm as lgb
        except ImportError:
            raise RuntimeError("lightgbm not installed. Run: pip install lightgbm")
        self._lgb = lgb

        default_params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [3, 5, 10],
            'learning_rate': 0.05,
            'num_leaves': 31,
            'min_data_in_leaf': 50,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'verbose': -1,
            'random_state': random_state,
        }
        if params:
            default_params.update(params)
        self.params = default_params
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.model: object | None = None
        self.feature_cols: list[str] = []
        self._cat_cols: list[str] = []

    # ------------------------------------------------------------------ FIT

    def fit(
        self,
        train: pd.DataFrame,
        valid: pd.DataFrame | None = None,
        verbose: bool = False,
    ) -> 'LightGBMRanker':
        """Fit on a panel DataFrame (must include 'date', 'target_relevance')."""
        train = train.dropna(subset=['target_relevance', 'target_fwd_return']).copy()
        if train.empty:
            raise ValueError("Empty training data after dropping NaN targets.")

        self.feature_cols = get_feature_columns(train)

        # Handle categoricals (sector)
        cat_features = []
        if 'sector' in self.feature_cols:
            train['sector'] = train['sector'].astype('category')
            cat_features.append('sector')
            if valid is not None:
                valid = valid.copy()
                valid['sector'] = valid['sector'].astype('category')
        self._cat_cols = cat_features

        # Build groups (one per date)
        train = train.sort_values(['date', 'ticker'])
        train_groups = train.groupby('date').size().values

        X_train = train[self.feature_cols]
        y_train = train['target_relevance'].astype(int)

        train_set = self._lgb.Dataset(
            X_train, label=y_train, group=train_groups,
            categorical_feature=cat_features,
        )

        valid_sets = [train_set]
        valid_names = ['train']
        callbacks = []
        if valid is not None and not valid.empty:
            valid = valid.dropna(subset=['target_relevance', 'target_fwd_return'])
            valid = valid.sort_values(['date', 'ticker'])
            valid_groups = valid.groupby('date').size().values
            valid_set = self._lgb.Dataset(
                valid[self.feature_cols],
                label=valid['target_relevance'].astype(int),
                group=valid_groups,
                categorical_feature=cat_features,
                reference=train_set,
            )
            valid_sets.append(valid_set)
            valid_names.append('valid')
            callbacks.append(self._lgb.early_stopping(self.early_stopping_rounds, verbose=verbose))
        if verbose:
            callbacks.append(self._lgb.log_evaluation(period=50))

        self.model = self._lgb.train(
            params=self.params,
            train_set=train_set,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        return self

    # ------------------------------------------------------------- PREDICT

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Predict ranking score (higher = more likely top fwd-return)."""
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        df = df.copy()
        if 'sector' in self.feature_cols and 'sector' in df.columns:
            df['sector'] = df['sector'].astype('category')
        X = df[self.feature_cols]
        scores = self.model.predict(X, num_iteration=self.model.best_iteration)
        return pd.Series(scores, index=df.index, name='score')

    # ------------------------------------------------ FEATURE IMPORTANCE

    def feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """Return feature importance as DataFrame sorted desc."""
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        imp = self.model.feature_importance(importance_type=importance_type)
        return pd.DataFrame({
            'feature': self.feature_cols,
            'importance': imp,
        }).sort_values('importance', ascending=False).reset_index(drop=True)

    # ------------------------------------------------------- PERSISTENCE

    def save(self, path: str | Path):
        """Save model to disk (LightGBM native format + metadata)."""
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'params': self.params,
            'feature_cols': self.feature_cols,
            'num_boost_round': self.num_boost_round,
            'early_stopping_rounds': self.early_stopping_rounds,
            'cat_cols': self._cat_cols,
        }, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> 'LightGBMRanker':
        import joblib
        bundle = joblib.load(path)
        obj = cls(
            params=bundle['params'],
            num_boost_round=bundle['num_boost_round'],
            early_stopping_rounds=bundle['early_stopping_rounds'],
        )
        obj.model = bundle['model']
        obj.feature_cols = bundle['feature_cols']
        obj._cat_cols = bundle.get('cat_cols', [])
        return obj
