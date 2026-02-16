"""
Utility helpers for AutoFE-PG.

Functions for GPU detection, task inference, default metrics, and score comparison.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, mean_squared_error


def detect_gpu() -> bool:
    """Check if XGBoost can use GPU acceleration.

    Returns
    -------
    bool
        True if GPU is available and functional for XGBoost.
    """
    try:
        tmp = xgb.XGBClassifier(
            tree_method="hist",
            device="cuda",
            n_estimators=1,
            verbosity=0,
        )
        tmp.fit(np.array([[1, 2], [3, 4]]), np.array([0, 1]))
        return True
    except Exception:
        return False


def infer_task(y: pd.Series) -> str:
    """Infer whether the task is classification or regression.

    Parameters
    ----------
    y : pd.Series
        Target variable.

    Returns
    -------
    str
        ``'classification'`` or ``'regression'``.
    """
    if y.dtype == "object" or y.dtype.name == "category":
        return "classification"
    n_unique = y.nunique()
    if n_unique <= 30 and np.all(y.dropna() == y.dropna().astype(int)):
        return "classification"
    return "regression"


def default_metric(task: str):
    """Return the default metric function and optimization direction.

    Parameters
    ----------
    task : str
        ``'classification'`` or ``'regression'``.

    Returns
    -------
    tuple
        ``(metric_function, direction)`` where direction is
        ``'maximize'`` or ``'minimize'``.
    """
    if task == "classification":
        return roc_auc_score, "maximize"
    else:
        return mean_squared_error, "minimize"


def is_improvement(
    new_score: float,
    old_score: float,
    direction: str,
    threshold: float = 1e-7,
) -> bool:
    """Check whether ``new_score`` improves over ``old_score``.

    Parameters
    ----------
    new_score : float
        Candidate score.
    old_score : float
        Current best score.
    direction : str
        ``'maximize'`` or ``'minimize'``.
    threshold : float
        Minimum improvement required.

    Returns
    -------
    bool
        True if ``new_score`` is better than ``old_score`` by at least ``threshold``.
    """
    if direction == "maximize":
        return new_score > old_score + threshold
    else:
        return new_score < old_score - threshold
