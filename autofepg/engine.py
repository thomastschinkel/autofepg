"""
XGBoost cross-validation engine for AutoFE-PG.

Provides a lightweight wrapper around XGBoost for K-fold evaluation,
returning both mean score and standard deviation across folds.
"""

import gc
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

from autofepg.utils import default_metric


class XGBCVEngine:
    """Lightweight XGBoost cross-validation engine.

    Parameters
    ----------
    task : str
        ``'classification'`` or ``'regression'``.
    n_folds : int
        Number of CV folds.
    random_state : int
        Random seed.
    use_gpu : bool
        Whether to use GPU acceleration.
    metric_fn : callable, optional
        Custom metric function ``(y_true, y_pred) -> float``.
    metric_direction : str, optional
        ``'maximize'`` or ``'minimize'``.
    xgb_params : dict, optional
        Custom XGBoost parameters.
    """

    def __init__(
        self,
        task: str = "classification",
        n_folds: int = 5,
        random_state: int = 42,
        use_gpu: bool = False,
        metric_fn=None,
        metric_direction: Optional[str] = None,
        xgb_params: Optional[Dict[str, Any]] = None,
    ):
        self.task = task
        self.n_folds = n_folds
        self.random_state = random_state
        self.use_gpu = use_gpu
        self.xgb_params = xgb_params

        if metric_fn is not None:
            self.metric_fn = metric_fn
            self.metric_direction = metric_direction or "maximize"
        else:
            self.metric_fn, self.metric_direction = default_metric(task)

    def get_fold_indices(
        self, X: pd.DataFrame, y: pd.Series
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate cross-validation fold indices.

        Parameters
        ----------
        X : pd.DataFrame
            Features.
        y : pd.Series
            Target.

        Returns
        -------
        list of (train_indices, val_indices) tuples
        """
        if self.task == "classification":
            kf = StratifiedKFold(
                n_splits=self.n_folds,
                shuffle=True,
                random_state=self.random_state,
            )
        else:
            kf = KFold(
                n_splits=self.n_folds,
                shuffle=True,
                random_state=self.random_state,
            )
        return list(kf.split(X, y))

    def _get_params(self) -> Dict[str, Any]:
        """Build XGBoost parameter dictionary."""
        if self.xgb_params is not None:
            params = dict(self.xgb_params)
            if self.use_gpu:
                params.setdefault("tree_method", "hist")
                params.setdefault("device", "cuda")
            else:
                params.setdefault("tree_method", "hist")
            params.setdefault("random_state", self.random_state)
            params.setdefault("verbosity", 0)
        else:
            params = {
                "n_estimators": 500,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "random_state": self.random_state,
                "verbosity": 0,
                "n_jobs": -1,
            }
            if self.use_gpu:
                params["tree_method"] = "hist"
                params["device"] = "cuda"
            else:
                params["tree_method"] = "hist"

        return params

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        fold_indices: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Tuple[float, float]:
        """Run K-fold cross-validation and return (mean_score, std_score).

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target variable.
        fold_indices : list of (train_idx, val_idx) tuples
            Pre-computed fold indices.

        Returns
        -------
        tuple of (float, float)
            ``(mean_score, std_score)`` across folds.
        """
        params = self._get_params()
        scores = []

        early_stopping_rounds = params.pop("early_stopping_rounds", 50)

        # Label-encode categoricals for XGBoost
        X_processed = X.copy()
        for c in X_processed.columns:
            if X_processed[c].dtype == "object" or X_processed[c].dtype.name == "category":
                le = LabelEncoder()
                X_processed[c] = le.fit_transform(
                    X_processed[c].fillna("__NAN__").astype(str)
                )

        X_processed = X_processed.fillna(-999)

        for c in X_processed.columns:
            if X_processed[c].dtype not in [
                np.float64,
                np.float32,
                np.int64,
                np.int32,
                np.int16,
                np.int8,
                np.float16,
                np.uint8,
            ]:
                try:
                    X_processed[c] = X_processed[c].astype(float)
                except Exception:
                    X_processed[c] = LabelEncoder().fit_transform(
                        X_processed[c].astype(str)
                    )

        for fold_idx, (tr_idx, va_idx) in enumerate(fold_indices):
            X_tr = X_processed.iloc[tr_idx]
            X_va = X_processed.iloc[va_idx]
            y_tr = y.iloc[tr_idx]
            y_va = y.iloc[va_idx]

            if self.task == "classification":
                n_classes = y.nunique()
                if n_classes == 2:
                    model = xgb.XGBClassifier(
                        **params,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        early_stopping_rounds=early_stopping_rounds,
                    )
                else:
                    model = xgb.XGBClassifier(
                        **params,
                        objective="multi:softprob",
                        num_class=n_classes,
                        eval_metric="mlogloss",
                        early_stopping_rounds=early_stopping_rounds,
                    )
            else:
                model = xgb.XGBRegressor(
                    **params,
                    objective="reg:squarederror",
                    eval_metric="rmse",
                    early_stopping_rounds=early_stopping_rounds,
                )

            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                verbose=False,
            )

            if self.task == "classification":
                n_classes = y.nunique()
                if n_classes == 2:
                    preds = model.predict_proba(X_va)[:, 1]
                else:
                    preds = model.predict_proba(X_va)
                try:
                    if n_classes == 2:
                        score = self.metric_fn(y_va, preds)
                    else:
                        score = self.metric_fn(y_va, preds, multi_class="ovr")
                except Exception:
                    preds_class = model.predict(X_va)
                    score = accuracy_score(y_va, preds_class)
            else:
                preds = model.predict(X_va)
                score = self.metric_fn(y_va, preds)

            scores.append(score)
            del model
            gc.collect()

        return float(np.mean(scores)), float(np.std(scores))
