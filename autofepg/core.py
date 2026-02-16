"""
Core AutoFE class and convenience function for AutoFE-PG.

Implements greedy forward feature selection with optional backward pruning,
using XGBoost cross-validation as the evaluation engine.
"""

import gc
import time
import warnings
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union

from sklearn.preprocessing import LabelEncoder

from autofepg.utils import detect_gpu, infer_task, is_improvement
from autofepg.generators import FeatureGenerator
from autofepg.builder import FeatureCandidateBuilder
from autofepg.engine import XGBCVEngine

warnings.filterwarnings("ignore")


class AutoFE:
    """Automatic Feature Engineering and Selection.

    Generates feature candidates, evaluates each via XGBoost K-fold CV,
    and greedily selects features that improve the score.

    Parameters
    ----------
    task : str
        ``'classification'``, ``'regression'``, or ``'auto'``.
    n_folds : int
        Number of CV folds.
    time_budget : float, optional
        Maximum wall-clock time in seconds.
    random_state : int
        Random seed.
    metric_fn : callable, optional
        Custom metric ``(y_true, y_pred) -> float``.
    metric_direction : str, optional
        ``'maximize'`` or ``'minimize'``.
    max_pair_cols : int
        Max columns for pairwise features.
    max_digit_positions : int
        Max digit positions to extract.
    max_digit_interaction_order : int
        Max order for digit interactions.
    rounding_decimals : list of int, optional
        Decimal places for rounding features.
    quantile_bins : list of int, optional
        Bin counts for quantile binning.
    verbose : bool
        Print progress.
    xgb_params : dict, optional
        Custom XGBoost hyperparameters.
    improvement_threshold : float
        Minimum score improvement to keep a feature.
    sample : int, optional
        Subsample rows for faster CV evaluation.
    backward_selection : bool
        Whether to run backward pruning after forward selection.

    Attributes
    ----------
    selected_generators_ : list of FeatureGenerator
        Generators selected during fit.
    base_score_ : float
        Baseline CV score (mean).
    base_score_std_ : float
        Baseline CV score standard deviation.
    best_score_ : float
        Best CV score after selection (mean).
    best_score_std_ : float
        Best CV score standard deviation.
    history_ : list of dict
        Full selection history.

    Examples
    --------
    >>> from autofepg import AutoFE
    >>> autofe = AutoFE(task="classification", time_budget=3600)
    >>> X_train_new, X_test_new = autofe.fit_select(
    ...     X_train, y_train, X_test,
    ...     aux_target_cols=["col_a", "col_b"],
    ... )
    >>> print(autofe.get_selected_feature_names())

    Custom XGBoost parameters:

    >>> autofe = AutoFE(
    ...     task="classification",
    ...     xgb_params={"n_estimators": 1000, "max_depth": 8, "learning_rate": 0.05},
    ... )

    Sampling for faster evaluation:

    >>> autofe = AutoFE(sample=10000)

    Backward selection:

    >>> autofe = AutoFE(backward_selection=True)
    """

    def __init__(
        self,
        task: str = "auto",
        n_folds: int = 5,
        time_budget: Optional[float] = None,
        random_state: int = 42,
        metric_fn=None,
        metric_direction: Optional[str] = None,
        max_pair_cols: int = 20,
        max_digit_positions: int = 4,
        max_digit_interaction_order: int = 3,
        rounding_decimals: Optional[List[int]] = None,
        quantile_bins: Optional[List[int]] = None,
        verbose: bool = True,
        xgb_params: Optional[Dict[str, Any]] = None,
        improvement_threshold: float = 1e-7,
        sample: Optional[int] = None,
        backward_selection: bool = False,
    ):
        self.task = task
        self.n_folds = n_folds
        self.time_budget = time_budget
        self.random_state = random_state
        self.metric_fn = metric_fn
        self.metric_direction = metric_direction
        self.max_pair_cols = max_pair_cols
        self.max_digit_positions = max_digit_positions
        self.max_digit_interaction_order = max_digit_interaction_order
        self.rounding_decimals = rounding_decimals
        self.quantile_bins = quantile_bins
        self.verbose = verbose
        self.xgb_params = xgb_params
        self.improvement_threshold = improvement_threshold
        self.sample = sample
        self.backward_selection = backward_selection

        self.selected_generators_: List[FeatureGenerator] = []
        self.base_score_: Optional[float] = None
        self.base_score_std_: Optional[float] = None
        self.best_score_: Optional[float] = None
        self.best_score_std_: Optional[float] = None
        self.history_: List[Dict[str, Any]] = []

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _sample_data(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Subsample data if ``self.sample`` is set and smaller than ``len(X)``."""
        if self.sample is not None and self.sample < len(X):
            if self.task == "classification":
                from sklearn.model_selection import train_test_split

                _, X_s, _, y_s = train_test_split(
                    X,
                    y,
                    test_size=self.sample,
                    random_state=self.random_state,
                    stratify=y,
                )
                return X_s.reset_index(drop=True), y_s.reset_index(drop=True)
            else:
                rng = np.random.RandomState(self.random_state)
                idx = rng.choice(len(X), size=self.sample, replace=False)
                return (
                    X.iloc[idx].reset_index(drop=True),
                    y.iloc[idx].reset_index(drop=True),
                )
        return X, y

    def _backward_feature_selection(
        self,
        X_base_eval: pd.DataFrame,
        y_eval: pd.Series,
        fold_indices_eval: List[Tuple[np.ndarray, np.ndarray]],
        engine: XGBCVEngine,
        start_time: float,
    ):
        """Try removing each selected feature; drop those that don't help."""
        if len(self.selected_generators_) == 0:
            self._log("[AutoFE-PG] No selected features to prune in backward selection.")
            return X_base_eval

        self._log(f"\n[AutoFE-PG] ========== BACKWARD FEATURE SELECTION ==========")
        self._log(
            f"[AutoFE-PG] Starting with {len(self.selected_generators_)} selected features."
        )
        self._log(
            f"[AutoFE-PG] Current best score: "
            f"{self.best_score_:.6f} ± {self.best_score_std_:.6f}"
        )

        # Build current feature matrix
        X_current = X_base_eval.copy()
        selected_feat_dfs: Dict[str, pd.DataFrame] = {}
        for gen in self.selected_generators_:
            try:
                feat_df = gen.fit_transform(
                    X_base_eval.copy(), y_eval, fold_indices_eval
                )
                selected_feat_dfs[gen.name] = feat_df
                X_current = pd.concat([X_current, feat_df], axis=1)
            except Exception as e:
                self._log(
                    f"[AutoFE-PG] Warning: could not regenerate {gen.name} "
                    f"during backward selection: {e}"
                )

        current_score = self.best_score_
        current_score_std = self.best_score_std_

        improved = True
        pass_num = 0

        while improved:
            improved = False
            pass_num += 1
            self._log(
                f"\n[AutoFE-PG] Backward pass {pass_num}, "
                f"{len(self.selected_generators_)} features remaining."
            )

            generators_to_remove = []

            for idx, gen in enumerate(self.selected_generators_):
                if self.time_budget is not None:
                    elapsed = time.time() - start_time
                    if elapsed > self.time_budget:
                        self._log(
                            "[AutoFE-PG] Time budget exhausted during backward "
                            "selection. Stopping."
                        )
                        return X_current

                if gen.name not in selected_feat_dfs:
                    continue

                cols_to_drop = list(selected_feat_dfs[gen.name].columns)
                X_trial = X_current.drop(columns=cols_to_drop, errors="ignore")

                try:
                    trial_mean, trial_std = engine.evaluate(
                        X_trial, y_eval, fold_indices_eval
                    )
                except Exception as e:
                    self._log(
                        f"    [{idx + 1}/{len(self.selected_generators_)}] "
                        f"Error evaluating without {gen.name}: {e}"
                    )
                    continue

                is_better = is_improvement(
                    trial_mean,
                    current_score,
                    self.metric_direction,
                    self.improvement_threshold,
                )

                if is_better:
                    self._log(
                        f"    [{idx + 1}/{len(self.selected_generators_)}] "
                        f"Remove {gen.name:<60s} "
                        f"score={trial_mean:.6f}±{trial_std:.6f} > "
                        f"current={current_score:.6f} ✓ REMOVE"
                    )
                    generators_to_remove.append(gen)
                    current_score = trial_mean
                    current_score_std = trial_std
                    improved = True
                else:
                    self._log(
                        f"    [{idx + 1}/{len(self.selected_generators_)}] "
                        f"Remove {gen.name:<60s} "
                        f"score={trial_mean:.6f}±{trial_std:.6f} ≤ "
                        f"current={current_score:.6f} ✗ KEEP"
                    )

            if generators_to_remove:
                for gen in generators_to_remove:
                    self.selected_generators_.remove(gen)
                    if gen.name in selected_feat_dfs:
                        cols_to_drop = list(selected_feat_dfs[gen.name].columns)
                        X_current = X_current.drop(
                            columns=cols_to_drop, errors="ignore"
                        )
                        del selected_feat_dfs[gen.name]

                self.best_score_ = current_score
                self.best_score_std_ = current_score_std
                self._log(
                    f"[AutoFE-PG] Removed {len(generators_to_remove)} feature(s) "
                    f"in pass {pass_num}. New best: "
                    f"{self.best_score_:.6f} ± {self.best_score_std_:.6f}"
                )

        self._log(
            f"[AutoFE-PG] Backward selection complete. "
            f"{len(self.selected_generators_)} features remaining."
        )
        self._log(
            f"[AutoFE-PG] Final score after backward: "
            f"{self.best_score_:.6f} ± {self.best_score_std_:.6f}"
        )

        return X_current

    def fit_select(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        cat_cols: Optional[List[str]] = None,
        num_cols: Optional[List[str]] = None,
        aux_target_cols: Optional[List[str]] = None,
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """Fit and greedily select features.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features.
        y_train : pd.Series
            Training target.
        X_test : pd.DataFrame, optional
            Test features. If provided, both augmented train and test are returned.
        cat_cols : list of str, optional
            Categorical columns. Auto-detected if None.
        num_cols : list of str, optional
            Numerical columns. Auto-detected if None.
        aux_target_cols : list of str, optional
            Columns to use as auxiliary targets for TE.

        Returns
        -------
        tuple of (pd.DataFrame, pd.DataFrame) or pd.DataFrame
            If ``X_test`` is provided: ``(X_train_augmented, X_test_augmented)``.
            Otherwise: ``X_train_augmented``.
        """
        start_time = time.time()

        # Infer task
        if self.task == "auto":
            self.task = infer_task(y_train)
            self._log(f"[AutoFE-PG] Inferred task: {self.task}")

        # Encode target if needed
        self._target_le = None
        y = y_train.copy()
        if self.task == "classification" and y.dtype == "object":
            self._target_le = LabelEncoder()
            y = pd.Series(self._target_le.fit_transform(y), index=y.index)

        # Sample data for evaluation
        X_eval, y_eval = self._sample_data(X_train, y)
        if self.sample is not None and self.sample < len(X_train):
            self._log(
                f"[AutoFE-PG] Using sample of {len(X_eval)} rows for evaluation "
                f"(full data: {len(X_train)} rows)"
            )

        # GPU check
        use_gpu = detect_gpu()
        self._log(f"[AutoFE-PG] GPU available: {use_gpu}")
        self._log(f"[AutoFE-PG] Improvement threshold: {self.improvement_threshold}")
        self._log(f"[AutoFE-PG] Backward selection: {self.backward_selection}")

        # Build engine
        engine = XGBCVEngine(
            task=self.task,
            n_folds=self.n_folds,
            random_state=self.random_state,
            use_gpu=use_gpu,
            metric_fn=self.metric_fn,
            metric_direction=self.metric_direction,
            xgb_params=self.xgb_params,
        )

        if self.metric_direction is None:
            self.metric_direction = engine.metric_direction

        # Fold indices
        fold_indices_eval = engine.get_fold_indices(X_eval, y_eval)
        fold_indices_full = engine.get_fold_indices(X_train, y)

        # Build candidates
        builder = FeatureCandidateBuilder(
            cat_cols=cat_cols,
            num_cols=num_cols,
            aux_target_cols=aux_target_cols or [],
            max_pair_cols=self.max_pair_cols,
            max_digit_positions=self.max_digit_positions,
            max_digit_interaction_order=self.max_digit_interaction_order,
            rounding_decimals=self.rounding_decimals,
            quantile_bins=self.quantile_bins,
        )
        candidates = builder.build(X_train, y)

        self._log(
            "\n[AutoFE-PG] Using hardcoded candidate sequence. "
            "Starting greedy selection.\n"
        )

        # Baseline score
        X_current_eval = X_eval.copy()
        self._log("[AutoFE-PG] Computing baseline score...")
        base_mean, base_std = engine.evaluate(
            X_current_eval, y_eval, fold_indices_eval
        )
        self.base_score_ = base_mean
        self.base_score_std_ = base_std
        self.best_score_ = base_mean
        self.best_score_std_ = base_std
        self._log(f"[AutoFE-PG] Baseline CV score: {base_mean:.6f} ± {base_std:.6f}")

        # Greedy forward selection
        self.selected_generators_ = []
        total_candidates = len(candidates)

        for i, gen in enumerate(candidates):
            elapsed = time.time() - start_time
            if self.time_budget is not None and elapsed > self.time_budget:
                self._log(
                    f"[AutoFE-PG] Time budget exhausted ({elapsed:.0f}s). Stopping."
                )
                break

            eta = ""
            if i > 0 and self.time_budget:
                rate = elapsed / i
                remaining = (total_candidates - i) * rate
                eta = f" | ETA: {remaining:.0f}s"

            try:
                new_feat_eval = gen.fit_transform(
                    X_current_eval, y_eval, fold_indices_eval
                )
                X_trial = pd.concat([X_current_eval, new_feat_eval], axis=1)
                trial_mean, trial_std = engine.evaluate(
                    X_trial, y_eval, fold_indices_eval
                )

                improved = is_improvement(
                    trial_mean,
                    self.best_score_,
                    self.metric_direction,
                    self.improvement_threshold,
                )

                status = "✓ KEEP" if improved else "✗ DROP"
                self._log(
                    f"[{i + 1}/{total_candidates}] {gen.name:<60s} "
                    f"score={trial_mean:.6f}±{trial_std:.6f} "
                    f"best={self.best_score_:.6f}±{self.best_score_std_:.6f} "
                    f"{status}{eta}"
                )

                self.history_.append(
                    {
                        "step": i + 1,
                        "name": gen.name,
                        "score_mean": trial_mean,
                        "score_std": trial_std,
                        "best_score_mean": self.best_score_,
                        "best_score_std": self.best_score_std_,
                        "kept": improved,
                        "elapsed": time.time() - start_time,
                    }
                )

                if improved:
                    self.best_score_ = trial_mean
                    self.best_score_std_ = trial_std
                    self.selected_generators_.append(gen)
                    X_current_eval = X_trial
                else:
                    del X_trial
                    gc.collect()

            except Exception as e:
                self._log(
                    f"[{i + 1}/{total_candidates}] {gen.name:<60s} "
                    f"ERROR: {str(e)[:80]}"
                )
                self.history_.append(
                    {
                        "step": i + 1,
                        "name": gen.name,
                        "score_mean": None,
                        "score_std": None,
                        "best_score_mean": self.best_score_,
                        "best_score_std": self.best_score_std_,
                        "kept": False,
                        "elapsed": time.time() - start_time,
                        "error": str(e),
                    }
                )

        elapsed_forward = time.time() - start_time
        self._log(f"\n[AutoFE-PG] ========== FORWARD SELECTION DONE ==========")
        self._log(
            f"[AutoFE-PG] Baseline score : "
            f"{self.base_score_:.6f} ± {self.base_score_std_:.6f}"
        )
        self._log(
            f"[AutoFE-PG] Best score     : "
            f"{self.best_score_:.6f} ± {self.best_score_std_:.6f}"
        )
        self._log(f"[AutoFE-PG] Features added : {len(self.selected_generators_)}")
        self._log(f"[AutoFE-PG] Forward time   : {elapsed_forward:.1f}s")

        # Backward selection
        if self.backward_selection:
            X_current_eval = self._backward_feature_selection(
                X_base_eval=X_eval,
                y_eval=y_eval,
                fold_indices_eval=fold_indices_eval,
                engine=engine,
                start_time=start_time,
            )

        elapsed_total = time.time() - start_time
        self._log(f"\n[AutoFE-PG] ========== DONE ==========")
        self._log(
            f"[AutoFE-PG] Baseline score : "
            f"{self.base_score_:.6f} ± {self.base_score_std_:.6f}"
        )
        self._log(
            f"[AutoFE-PG] Best score     : "
            f"{self.best_score_:.6f} ± {self.best_score_std_:.6f}"
        )
        self._log(f"[AutoFE-PG] Features added : {len(self.selected_generators_)}")
        self._log(f"[AutoFE-PG] Total time     : {elapsed_total:.1f}s")
        self._log(f"[AutoFE-PG] Selected features:")
        for g in self.selected_generators_:
            self._log(f"    - {g.name}")

        # Re-fit on full training data
        self._log(
            "[AutoFE-PG] Re-fitting selected generators on full training data..."
        )
        X_train_final = X_train.copy()
        for gen in self.selected_generators_:
            feat_df = gen.fit_transform(X_train_final, y, fold_indices_full)
            X_train_final = pd.concat([X_train_final, feat_df], axis=1)

        if X_test is not None:
            X_test_final = X_test.copy()
            for gen in self.selected_generators_:
                feat_df = gen.transform(X_test_final)
                X_test_final = pd.concat([X_test_final, feat_df], axis=1)
            return X_train_final, X_test_final

        return X_train_final

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply selected feature generators to new data.

        Parameters
        ----------
        X : pd.DataFrame
            Input features.

        Returns
        -------
        pd.DataFrame
            Augmented DataFrame.
        """
        X_out = X.copy()
        for gen in self.selected_generators_:
            feat_df = gen.transform(X_out)
            X_out = pd.concat([X_out, feat_df], axis=1)
        return X_out

    def get_selected_feature_names(self) -> List[str]:
        """Return names of selected engineered features."""
        return [g.name for g in self.selected_generators_]

    def get_history(self) -> pd.DataFrame:
        """Return selection history as a DataFrame."""
        return pd.DataFrame(self.history_)


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================


def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: Optional[pd.DataFrame] = None,
    task: str = "auto",
    cat_cols: Optional[List[str]] = None,
    num_cols: Optional[List[str]] = None,
    aux_target_cols: Optional[List[str]] = None,
    n_folds: int = 5,
    time_budget: Optional[float] = None,
    metric_fn=None,
    metric_direction: Optional[str] = None,
    random_state: int = 42,
    verbose: bool = True,
    max_pair_cols: int = 20,
    max_digit_positions: int = 4,
    xgb_params: Optional[Dict[str, Any]] = None,
    improvement_threshold: float = 1e-7,
    sample: Optional[int] = None,
    backward_selection: bool = False,
) -> Dict[str, Any]:
    """One-liner convenience function for automatic feature engineering.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    X_test : pd.DataFrame, optional
        Test features. If provided, transformed test set is returned.
    task : str
        ``'classification'``, ``'regression'``, or ``'auto'``.
    cat_cols : list of str, optional
        Categorical columns. Auto-detected if None.
    num_cols : list of str, optional
        Numerical columns. Auto-detected if None.
    aux_target_cols : list of str, optional
        Columns in ``X_train`` to use as auxiliary targets for TE.
    n_folds : int
        Number of CV folds.
    time_budget : float, optional
        Maximum time in seconds.
    metric_fn : callable, optional
        Custom metric ``(y_true, y_pred) -> float``.
    metric_direction : str, optional
        ``'maximize'`` or ``'minimize'``.
    random_state : int
        Random seed.
    verbose : bool
        Print progress.
    max_pair_cols : int
        Max columns to consider for pairwise interactions.
    max_digit_positions : int
        Max digit positions to extract.
    xgb_params : dict, optional
        Custom XGBoost parameters.
    improvement_threshold : float
        Minimum score improvement to keep a feature.
    sample : int, optional
        Subsample rows for faster CV evaluation.
    backward_selection : bool
        Run backward feature selection after forward selection.

    Returns
    -------
    dict
        Keys: ``'X_train'``, ``'X_test'`` (if provided), ``'autofe'``,
        ``'history'``, ``'selected_features'``, ``'base_score'``,
        ``'base_score_std'``, ``'best_score'``, ``'best_score_std'``.
    """
    autofe = AutoFE(
        task=task,
        n_folds=n_folds,
        time_budget=time_budget,
        random_state=random_state,
        metric_fn=metric_fn,
        metric_direction=metric_direction,
        max_pair_cols=max_pair_cols,
        max_digit_positions=max_digit_positions,
        verbose=verbose,
        xgb_params=xgb_params,
        improvement_threshold=improvement_threshold,
        sample=sample,
        backward_selection=backward_selection,
    )

    result = autofe.fit_select(
        X_train,
        y_train,
        X_test,
        cat_cols=cat_cols,
        num_cols=num_cols,
        aux_target_cols=aux_target_cols,
    )

    output: Dict[str, Any] = {
        "autofe": autofe,
        "history": autofe.get_history(),
        "selected_features": autofe.get_selected_feature_names(),
        "base_score": autofe.base_score_,
        "base_score_std": autofe.base_score_std_,
        "best_score": autofe.best_score_,
        "best_score_std": autofe.best_score_std_,
    }

    if X_test is not None:
        output["X_train"] = result[0]
        output["X_test"] = result[1]
    else:
        output["X_train"] = result

    return output
