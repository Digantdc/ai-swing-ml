"""XGBoost binary classifier for per-stock direction prediction.

Replaces the v1 LightGBM LambdaRank. Target:
    P(stock_return_in_21d > 3%)

Why XGBoost over LightGBM:
    - For binary classification on tabular features, both perform similarly;
      XGBoost has the more mature calibration API (isotonic regression wrapper).
    - We want CALIBRATED probabilities, not just rankings — the threshold
      logic in downstream strategy selection assumes P(up) is interpretable
      as an actual probability.

Why binary classification over ranking (v1 was Ranker):
    - "Will THIS stock go up?" is a learnable per-stock question.
    - "Rank these 60 correlated AI stocks" is not — they all move together.
    - Confirmed empirically: v1 had Rank IC -0.017 (worse than random).

Walk-forward training is supported via the .fit() / .predict_proba() interface.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


META_COLS = {
    'ticker', 'date', 'close', 'sector',
    'target_fwd_return', 'target_fwd_vol',
    'target_rank_pct', 'target_top_pct', 'target_relevance',
    'target_up_in_21d', 'target_up_3pct',
    'regime', 'regime_score',
}


class BinaryDirectionClassifier:
    """XGBoost binary classifier with optional probability calibration."""

    def __init__(
        self,
        params: dict | None = None,
        num_boost_round: int = 500,
        early_stopping_rounds: int = 50,
        random_state: int = 42,
        calibrate: bool = True,
        feature_subset: list[str] | None = None,
    ):
        try:
            import xgboost as xgb
        except ImportError:
            raise RuntimeError("xgboost not installed. Run: pip install xgboost")
        self._xgb = xgb

        default_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_child_weight': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'tree_method': 'hist',
            'random_state': random_state,
        }
        if params:
            default_params.update(params)
        self.params = default_params
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.calibrate = calibrate
        self.feature_subset = feature_subset

        self.model = None
        self.calibrator = None
        self.feature_cols: list[str] = []
        self._cat_cols: list[str] = []

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def make_target(
        df: pd.DataFrame,
        threshold: float = 0.03,
    ) -> pd.Series:
        """Build the binary target column from continuous forward return.

        target = 1 if forward 21d return > threshold (e.g., 3%), else 0.
        Stocks with neither up >3% nor down (just flat) are still labeled 0.
        """
        return (df['target_fwd_return'] > threshold).astype(int)

    def _resolve_features(self, df: pd.DataFrame) -> list[str]:
        if self.feature_subset is not None:
            return [c for c in self.feature_subset if c in df.columns]
        return [c for c in df.columns if c not in META_COLS]

    # ------------------------------------------------------------------ FIT

    def fit(
        self,
        train: pd.DataFrame,
        valid: pd.DataFrame | None = None,
        target_threshold: float = 0.03,
        verbose: bool = False,
    ) -> 'BinaryDirectionClassifier':
        train = train.dropna(subset=['target_fwd_return']).copy()
        if train.empty:
            raise ValueError("Empty training data.")

        train['_y'] = self.make_target(train, target_threshold)
        if train['_y'].nunique() < 2:
            raise ValueError("Target has only one class — adjust threshold.")

        self.feature_cols = self._resolve_features(train)

        # Categorical handling: convert 'sector' if present
        cat_features = []
        if 'sector' in self.feature_cols:
            train['sector'] = train['sector'].astype('category').cat.codes
            cat_features.append('sector')
            if valid is not None:
                valid = valid.copy()
                valid['sector'] = valid['sector'].astype('category').cat.codes
        self._cat_cols = cat_features

        X_train = train[self.feature_cols]
        y_train = train['_y']
        dtrain = self._xgb.DMatrix(X_train, label=y_train, missing=np.nan)

        evals = [(dtrain, 'train')]
        if valid is not None and not valid.empty:
            valid = valid.dropna(subset=['target_fwd_return']).copy()
            valid['_y'] = self.make_target(valid, target_threshold)
            dvalid = self._xgb.DMatrix(
                valid[self.feature_cols], label=valid['_y'], missing=np.nan,
            )
            evals.append((dvalid, 'valid'))

        self.model = self._xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=self.num_boost_round,
            evals=evals,
            early_stopping_rounds=self.early_stopping_rounds if valid is not None else None,
            verbose_eval=verbose,
        )

        # Calibrate probabilities via isotonic regression on validation set
        if self.calibrate and valid is not None and not valid.empty:
            try:
                from sklearn.isotonic import IsotonicRegression
                raw_probs = self.model.predict(dvalid)
                self.calibrator = IsotonicRegression(out_of_bounds='clip')
                self.calibrator.fit(raw_probs, valid['_y'].values)
                logger.info("Probability calibration fitted via isotonic regression.")
            except Exception as e:
                logger.warning(f"Calibration failed (raw probs will be used): {e}")
                self.calibrator = None
        return self

    # ------------------------------------------------------------ PREDICT

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        """Calibrated P(stock_up_3pct_in_21d) per row."""
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        df = df.copy()
        if 'sector' in self.feature_cols and 'sector' in df.columns:
            df['sector'] = df['sector'].astype('category').cat.codes
        missing = [c for c in self.feature_cols if c not in df.columns]
        for c in missing:
            df[c] = np.nan
        X = df[self.feature_cols]
        dmat = self._xgb.DMatrix(X, missing=np.nan)
        probs = self.model.predict(dmat)
        if self.calibrator is not None:
            probs = self.calibrator.transform(probs)
        return pd.Series(probs, index=df.index, name='p_up')

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
        """Binary 0/1 prediction at given probability threshold."""
        return (self.predict_proba(df) >= threshold).astype(int)

    # ------------------------------------------------ FEATURE IMPORTANCE

    def feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        score = self.model.get_score(importance_type=importance_type)
        rows = []
        for feat in self.feature_cols:
            rows.append({'feature': feat, 'importance': score.get(feat, 0.0)})
        return pd.DataFrame(rows).sort_values('importance', ascending=False).reset_index(drop=True)

    # ------------------------------------------------------- PERSISTENCE

    def save(self, path: str | Path):
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model_bytes': self.model.save_raw() if self.model else None,
            'params': self.params,
            'feature_cols': self.feature_cols,
            'num_boost_round': self.num_boost_round,
            'early_stopping_rounds': self.early_stopping_rounds,
            'calibrator': self.calibrator,
            'feature_subset': self.feature_subset,
            'cat_cols': self._cat_cols,
        }, path)
        logger.info(f"Classifier saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> 'BinaryDirectionClassifier':
        import joblib
        bundle = joblib.load(path)
        obj = cls(
            params=bundle['params'],
            num_boost_round=bundle['num_boost_round'],
            early_stopping_rounds=bundle['early_stopping_rounds'],
            feature_subset=bundle.get('feature_subset'),
        )
        import xgboost as xgb
        booster = xgb.Booster()
        booster.load_model(bytearray(bundle['model_bytes']))
        obj.model = booster
        obj.feature_cols = bundle['feature_cols']
        obj.calibrator = bundle.get('calibrator')
        obj._cat_cols = bundle.get('cat_cols', [])
        return obj
