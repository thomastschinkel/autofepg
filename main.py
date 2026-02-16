"""
AutoFE - Automatic Feature Engineering Library for Kaggle Playground Competitions
=================================================================================

A complete, production-ready library that:
- Auto-detects categorical/numerical columns
- Generates a comprehensive set of feature engineering ideas
- Greedily selects features one-by-one based on CV improvement
- Supports classification and regression
- Runs XGBoost on GPU if available
- Respects a time budget
- Allows custom XGBoost parameters
- No target leakage in any feature engineering step
"""

import numpy as np
import pandas as pd
import time
import warnings
import gc
from itertools import combinations
from typing import List, Optional, Tuple, Dict, Any, Union

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

warnings.filterwarnings("ignore")


# ============================================================================
# UTILITY HELPERS
# ============================================================================

def _detect_gpu() -> bool:
    """Check if XGBoost can use GPU."""
    try:
        _tmp = xgb.XGBClassifier(
            tree_method="hist", device="cuda",
            n_estimators=1, verbosity=0
        )
        _tmp.fit(np.array([[1, 2], [3, 4]]), np.array([0, 1]))
        return True
    except Exception:
        return False


def _infer_task(y: pd.Series) -> str:
    """Infer whether the task is 'classification' or 'regression'."""
    if y.dtype == "object" or y.dtype.name == "category":
        return "classification"
    n_unique = y.nunique()
    if n_unique <= 30 and np.all(y.dropna() == y.dropna().astype(int)):
        return "classification"
    return "regression"


def _default_metric(task: str):
    """Return (metric_function, direction) where direction is 'maximize' or 'minimize'."""
    if task == "classification":
        return roc_auc_score, "maximize"
    else:
        return mean_squared_error, "minimize"


def _is_improvement(new_score: float, old_score: float, direction: str) -> bool:
    if direction == "maximize":
        return new_score > old_score + 1e-7
    else:
        return new_score < old_score - 1e-7


# ============================================================================
# FEATURE GENERATORS
# ============================================================================

class FeatureGenerator:
    """
    Each feature generator produces one or more columns.
    It implements:
        - fit_transform(df_train, y_train, folds) -> pd.DataFrame of new columns for train
        - transform(df_test) -> pd.DataFrame of new columns for test
        - name: a human-readable name

    IMPORTANT: All generators that use the target (y) MUST do so in an
    out-of-fold manner to prevent target leakage. The full-data mapping
    used at transform time is built from the entire training set AFTER
    selection is complete, which is acceptable because test data never
    leaks into training evaluation.
    """

    def __init__(self, name: str):
        self.name = name

    def fit_transform(self, df: pd.DataFrame, y: pd.Series,
                      fold_indices: List[Tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
        raise NotImplementedError

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 1) Pairwise Combination of Base Features (cat x cat, cat x num, num x num)
# ---------------------------------------------------------------------------

class PairInteraction(FeatureGenerator):
    """Create a string interaction feature between two columns."""

    def __init__(self, col_a: str, col_b: str):
        super().__init__(f"pair__{col_a}_x_{col_b}")
        self.col_a = col_a
        self.col_b = col_b

    def fit_transform(self, df, y, fold_indices):
        s = df[self.col_a].astype(str) + "__" + df[self.col_b].astype(str)
        self._le = LabelEncoder()
        vals = self._le.fit_transform(s.fillna("__NAN__"))
        return pd.DataFrame({self.name: vals}, index=df.index)

    def transform(self, df):
        s = df[self.col_a].astype(str) + "__" + df[self.col_b].astype(str)
        s = s.fillna("__NAN__")
        le_classes = set(self._le.classes_)
        mapped = []
        for v in s:
            if v in le_classes:
                mapped.append(self._le.transform([v])[0])
            else:
                mapped.append(-1)
        return pd.DataFrame({self.name: mapped}, index=df.index)


# ---------------------------------------------------------------------------
# 2) Target Encoding (OUT-OF-FOLD only — no leakage)
# ---------------------------------------------------------------------------

class TargetEncoding(FeatureGenerator):
    """
    Out-of-fold target encoding for a single column.

    During fit_transform:
      - For each validation fold, the encoding is computed ONLY from the
        corresponding training fold. This prevents target leakage.
      - A full-data mapping is stored for use at test time only.

    During transform (test data):
      - The full-data mapping is applied. This is safe because the test
        target is never seen.
    """

    def __init__(self, col: str, smoothing: float = 20.0, suffix: str = ""):
        nm = f"te__{col}" if not suffix else f"te__{col}_{suffix}"
        super().__init__(nm)
        self.col = col
        self.smoothing = smoothing
        self.global_mean_ = None
        self.mapping_ = None

    def fit_transform(self, df, y, fold_indices):
        self.global_mean_ = y.mean()
        result = pd.Series(np.nan, index=df.index, dtype=float)

        for fold_idx, (tr_idx, va_idx) in enumerate(fold_indices):
            tr_col = df[self.col].iloc[tr_idx]
            tr_y = y.iloc[tr_idx]
            stats = tr_y.groupby(tr_col).agg(["mean", "count"])
            smooth = (stats["count"] * stats["mean"] + self.smoothing * self.global_mean_) / \
                     (stats["count"] + self.smoothing)
            result.iloc[va_idx] = df[self.col].iloc[va_idx].map(smooth)

        # Full mapping for test (uses all training data — no test leakage)
        full_stats = y.groupby(df[self.col]).agg(["mean", "count"])
        self.mapping_ = (full_stats["count"] * full_stats["mean"] + self.smoothing * self.global_mean_) / \
                        (full_stats["count"] + self.smoothing)

        result = result.fillna(self.global_mean_)
        return pd.DataFrame({self.name: result.values}, index=df.index)

    def transform(self, df):
        vals = df[self.col].map(self.mapping_).fillna(self.global_mean_)
        return pd.DataFrame({self.name: vals.values}, index=df.index)


# ---------------------------------------------------------------------------
# 3) Count Encoding (no target used — no leakage concern)
# ---------------------------------------------------------------------------

class CountEncoding(FeatureGenerator):
    """Count encoding for a single column. Does not use target — no leakage."""

    def __init__(self, col: str, suffix: str = ""):
        nm = f"ce__{col}" if not suffix else f"ce__{col}_{suffix}"
        super().__init__(nm)
        self.col = col
        self.mapping_ = None

    def fit_transform(self, df, y, fold_indices):
        self.mapping_ = df[self.col].value_counts()
        vals = df[self.col].map(self.mapping_).fillna(0)
        return pd.DataFrame({self.name: vals.values}, index=df.index)

    def transform(self, df):
        vals = df[self.col].map(self.mapping_).fillna(0)
        return pd.DataFrame({self.name: vals.values}, index=df.index)


# ---------------------------------------------------------------------------
# 4) Digit Extraction from Numerical Features (no target — no leakage)
# ---------------------------------------------------------------------------

class DigitFeature(FeatureGenerator):
    """Extract the i-th digit of a numerical feature (from its integer representation)."""

    def __init__(self, col: str, digit_pos: int):
        super().__init__(f"digit__{col}_pos{digit_pos}")
        self.col = col
        self.digit_pos = digit_pos

    def _extract(self, series: pd.Series) -> pd.Series:
        abs_int = series.fillna(0).abs().astype(np.int64).astype(str)

        def get_digit(s, pos):
            if pos < len(s):
                return int(s[-(pos + 1)])
            return 0

        return abs_int.apply(lambda x: get_digit(x, self.digit_pos))

    def fit_transform(self, df, y, fold_indices):
        vals = self._extract(df[self.col])
        return pd.DataFrame({self.name: vals.values}, index=df.index)

    def transform(self, df):
        vals = self._extract(df[self.col])
        return pd.DataFrame({self.name: vals.values}, index=df.index)


# ---------------------------------------------------------------------------
# 5) Digit Interaction (no target — no leakage)
# ---------------------------------------------------------------------------

class DigitInteraction(FeatureGenerator):
    """
    Interaction between digits of numerical features.
    E.g., digit0 of col_a * 10 + digit0 of col_b  (a simple hash).
    """

    def __init__(self, col_digit_pairs: List[Tuple[str, int]]):
        name_parts = "_".join([f"{c}d{d}" for c, d in col_digit_pairs])
        super().__init__(f"digitint__{name_parts}")
        self.col_digit_pairs = col_digit_pairs

    def _extract_digit(self, series: pd.Series, pos: int) -> pd.Series:
        abs_int = series.fillna(0).abs().astype(np.int64).astype(str)

        def get_digit(s, p):
            if p < len(s):
                return int(s[-(p + 1)])
            return 0

        return abs_int.apply(lambda x: get_digit(x, pos))

    def _compute(self, df: pd.DataFrame) -> pd.Series:
        result = pd.Series(np.zeros(len(df), dtype=np.int64), index=df.index)
        for i, (col, dpos) in enumerate(self.col_digit_pairs):
            digits = self._extract_digit(df[col], dpos)
            result = result * 10 + digits.values
        return result

    def fit_transform(self, df, y, fold_indices):
        vals = self._compute(df)
        return pd.DataFrame({self.name: vals.values}, index=df.index)

    def transform(self, df):
        vals = self._compute(df)
        return pd.DataFrame({self.name: vals.values}, index=df.index)


# ---------------------------------------------------------------------------
# 6) Rounding Features (no target — no leakage)
# ---------------------------------------------------------------------------

class RoundFeature(FeatureGenerator):
    """Round a numerical feature to a given number of decimals or magnitude."""

    def __init__(self, col: str, decimals: int):
        super().__init__(f"round__{col}_dec{decimals}")
        self.col = col
        self.decimals = decimals

    def fit_transform(self, df, y, fold_indices):
        vals = df[self.col].fillna(0).round(self.decimals)
        return pd.DataFrame({self.name: vals.values}, index=df.index)

    def transform(self, df):
        vals = df[self.col].fillna(0).round(self.decimals)
        return pd.DataFrame({self.name: vals.values}, index=df.index)


# ---------------------------------------------------------------------------
# 7) Quantile Binning (no target — no leakage)
# ---------------------------------------------------------------------------

class QuantileBinFeature(FeatureGenerator):
    """Bin a numerical column into quantile bins."""

    def __init__(self, col: str, n_bins: int = 10):
        super().__init__(f"qbin__{col}_b{n_bins}")
        self.col = col
        self.n_bins = n_bins
        self.bin_edges_ = None
        self.fill_median_ = None

    def fit_transform(self, df, y, fold_indices):
        self.fill_median_ = df[self.col].median()
        vals = df[self.col].fillna(self.fill_median_)
        try:
            binned, edges = pd.qcut(vals, q=self.n_bins, labels=False,
                                    retbins=True, duplicates="drop")
        except Exception:
            binned = pd.Series(0, index=df.index)
            edges = None
        self.bin_edges_ = edges
        return pd.DataFrame({self.name: binned.values}, index=df.index)

    def transform(self, df):
        vals = df[self.col].fillna(self.fill_median_ if self.fill_median_ is not None else 0)
        if self.bin_edges_ is not None:
            binned = pd.cut(vals, bins=self.bin_edges_, labels=False,
                            include_lowest=True)
            binned = binned.fillna(0).astype(int)
        else:
            binned = pd.Series(0, index=df.index)
        return pd.DataFrame({self.name: binned.values}, index=df.index)


# ---------------------------------------------------------------------------
# 8) Numeric-to-Categorical (no target — no leakage)
# ---------------------------------------------------------------------------

class NumToCat(FeatureGenerator):
    """Convert a numerical column into a categorical one via equal-width binning."""

    def __init__(self, col: str, n_bins: int = 5):
        super().__init__(f"num2cat__{col}_b{n_bins}")
        self.col = col
        self.n_bins = n_bins
        self.bin_edges_ = None
        self.fill_median_ = None

    def fit_transform(self, df, y, fold_indices):
        self.fill_median_ = df[self.col].median()
        vals = df[self.col].fillna(self.fill_median_)
        try:
            binned, edges = pd.cut(vals, bins=self.n_bins, labels=False,
                                   retbins=True, duplicates="drop")
        except Exception:
            binned = pd.Series(0, index=df.index)
            edges = None
        self.bin_edges_ = edges
        return pd.DataFrame({self.name: binned.fillna(0).astype(int).values}, index=df.index)

    def transform(self, df):
        vals = df[self.col].fillna(self.fill_median_ if self.fill_median_ is not None else 0)
        if self.bin_edges_ is not None:
            binned = pd.cut(vals, bins=self.bin_edges_, labels=False, include_lowest=True)
            binned = binned.fillna(0).astype(int)
        else:
            binned = pd.Series(0, index=df.index)
        return pd.DataFrame({self.name: binned.values}, index=df.index)


# ---------------------------------------------------------------------------
# 9) Target Encoding with an auxiliary target column (OUT-OF-FOLD — no leakage)
# ---------------------------------------------------------------------------

class TargetEncodingAuxTarget(FeatureGenerator):
    """
    Target-encode a column using a different column as the 'target'.

    The auxiliary target column is a feature column (e.g., 'employment_status'),
    NOT the actual prediction target. However, since we are encoding one feature
    using another feature's values, and both are available in train and test,
    there is no target leakage. We still use OOF encoding for consistency and
    to prevent overfitting.
    """

    def __init__(self, col: str, aux_target_col: str, smoothing: float = 20.0):
        super().__init__(f"te_aux__{col}_by_{aux_target_col}")
        self.col = col
        self.aux_target_col = aux_target_col
        self.smoothing = smoothing
        self.global_mean_ = None
        self.mapping_ = None
        self._aux_le = None

    def fit_transform(self, df, y, fold_indices):
        if self.aux_target_col not in df.columns:
            return pd.DataFrame({self.name: np.zeros(len(df))}, index=df.index)

        aux = df[self.aux_target_col].copy()
        # If aux is categorical, label-encode it
        if aux.dtype == "object" or aux.dtype.name == "category":
            self._aux_le = LabelEncoder()
            aux = pd.Series(self._aux_le.fit_transform(aux.fillna("__NAN__").astype(str)),
                            index=df.index, dtype=float)
        else:
            aux = aux.fillna(aux.mean())

        self.global_mean_ = aux.mean()
        result = pd.Series(np.nan, index=df.index, dtype=float)

        # OOF encoding to prevent overfitting
        for tr_idx, va_idx in fold_indices:
            tr_col = df[self.col].iloc[tr_idx]
            tr_aux = aux.iloc[tr_idx]
            stats = tr_aux.groupby(tr_col).agg(["mean", "count"])
            smooth = (stats["count"] * stats["mean"] + self.smoothing * self.global_mean_) / \
                     (stats["count"] + self.smoothing)
            result.iloc[va_idx] = df[self.col].iloc[va_idx].map(smooth)

        full_stats = aux.groupby(df[self.col]).agg(["mean", "count"])
        self.mapping_ = (full_stats["count"] * full_stats["mean"] + self.smoothing * self.global_mean_) / \
                        (full_stats["count"] + self.smoothing)

        result = result.fillna(self.global_mean_)
        return pd.DataFrame({self.name: result.values}, index=df.index)

    def transform(self, df):
        if self.mapping_ is None:
            return pd.DataFrame({self.name: np.zeros(len(df))}, index=df.index)
        vals = df[self.col].map(self.mapping_).fillna(self.global_mean_)
        return pd.DataFrame({self.name: vals.values}, index=df.index)


# ---------------------------------------------------------------------------
# 10) Arithmetic Interactions for Numericals (no target — no leakage)
# ---------------------------------------------------------------------------

class ArithmeticInteraction(FeatureGenerator):
    """sum, diff, product, ratio between two numerical columns."""

    def __init__(self, col_a: str, col_b: str, op: str = "add"):
        super().__init__(f"arith__{col_a}_{op}_{col_b}")
        self.col_a = col_a
        self.col_b = col_b
        self.op = op

    def _compute(self, df):
        a = df[self.col_a].fillna(0).astype(float)
        b = df[self.col_b].fillna(0).astype(float)
        if self.op == "add":
            return a + b
        elif self.op == "sub":
            return a - b
        elif self.op == "mul":
            return a * b
        elif self.op == "div":
            return a / (b + 1e-8)
        else:
            return a + b

    def fit_transform(self, df, y, fold_indices):
        vals = self._compute(df)
        return pd.DataFrame({self.name: vals.values}, index=df.index)

    def transform(self, df):
        vals = self._compute(df)
        return pd.DataFrame({self.name: vals.values}, index=df.index)


# ---------------------------------------------------------------------------
# 11) Frequency Encoding (no target — no leakage)
# ---------------------------------------------------------------------------

class FrequencyEncoding(FeatureGenerator):
    """Frequency (normalized count) encoding for a column."""

    def __init__(self, col: str):
        super().__init__(f"freq__{col}")
        self.col = col
        self.mapping_ = None

    def fit_transform(self, df, y, fold_indices):
        vc = df[self.col].value_counts(normalize=True)
        self.mapping_ = vc
        vals = df[self.col].map(vc).fillna(0)
        return pd.DataFrame({self.name: vals.values}, index=df.index)

    def transform(self, df):
        vals = df[self.col].map(self.mapping_).fillna(0)
        return pd.DataFrame({self.name: vals.values}, index=df.index)


# ============================================================================
# COMPOSITE FEATURE GENERATORS (TE/CE on derived columns)
# ============================================================================

class TargetEncodingOnPair(FeatureGenerator):
    """Target encoding on a pair interaction (creates the pair internally). OOF — no leakage."""

    def __init__(self, col_a: str, col_b: str, smoothing: float = 20.0):
        super().__init__(f"te__pair_{col_a}_x_{col_b}")
        self.col_a = col_a
        self.col_b = col_b
        self.smoothing = smoothing
        self.te_ = None

    def _make_pair_col(self, df):
        return df[self.col_a].astype(str) + "__" + df[self.col_b].astype(str)

    def fit_transform(self, df, y, fold_indices):
        pair_col = self._make_pair_col(df)
        tmp_df = df.copy()
        tmp_df["__pair__"] = pair_col
        self.te_ = TargetEncoding("__pair__", self.smoothing, suffix=f"pair_{self.col_a}_{self.col_b}")
        self.te_.name = self.name
        result = self.te_.fit_transform(tmp_df, y, fold_indices)
        result.columns = [self.name]
        return result

    def transform(self, df):
        tmp_df = df.copy()
        tmp_df["__pair__"] = self._make_pair_col(df)
        result = self.te_.transform(tmp_df)
        result.columns = [self.name]
        return result


class CountEncodingOnPair(FeatureGenerator):
    """Count encoding on a pair interaction. No target used — no leakage."""

    def __init__(self, col_a: str, col_b: str):
        super().__init__(f"ce__pair_{col_a}_x_{col_b}")
        self.col_a = col_a
        self.col_b = col_b
        self.ce_ = None

    def _make_pair_col(self, df):
        return df[self.col_a].astype(str) + "__" + df[self.col_b].astype(str)

    def fit_transform(self, df, y, fold_indices):
        pair_col = self._make_pair_col(df)
        tmp_df = df.copy()
        tmp_df["__pair__"] = pair_col
        self.ce_ = CountEncoding("__pair__", suffix=f"pair_{self.col_a}_{self.col_b}")
        self.ce_.name = self.name
        result = self.ce_.fit_transform(tmp_df, y, fold_indices)
        result.columns = [self.name]
        return result

    def transform(self, df):
        tmp_df = df.copy()
        tmp_df["__pair__"] = self._make_pair_col(df)
        result = self.ce_.transform(tmp_df)
        result.columns = [self.name]
        return result


class TargetEncodingOnDigit(FeatureGenerator):
    """Target encoding on a digit feature. OOF — no leakage."""

    def __init__(self, col: str, digit_pos: int, smoothing: float = 20.0):
        super().__init__(f"te__digit_{col}_pos{digit_pos}")
        self.col = col
        self.digit_pos = digit_pos
        self.smoothing = smoothing
        self.digit_gen_ = None
        self.te_ = None

    def fit_transform(self, df, y, fold_indices):
        self.digit_gen_ = DigitFeature(self.col, self.digit_pos)
        digit_df = self.digit_gen_.fit_transform(df, y, fold_indices)
        digit_col_name = self.digit_gen_.name

        tmp_df = df.copy()
        tmp_df[digit_col_name] = digit_df[digit_col_name]

        self.te_ = TargetEncoding(digit_col_name, self.smoothing)
        self.te_.name = self.name
        result = self.te_.fit_transform(tmp_df, y, fold_indices)
        result.columns = [self.name]
        return result

    def transform(self, df):
        digit_df = self.digit_gen_.transform(df)
        digit_col_name = self.digit_gen_.name
        tmp_df = df.copy()
        tmp_df[digit_col_name] = digit_df[digit_col_name]
        result = self.te_.transform(tmp_df)
        result.columns = [self.name]
        return result


class CountEncodingOnDigit(FeatureGenerator):
    """Count encoding on a digit feature. No target used — no leakage."""

    def __init__(self, col: str, digit_pos: int):
        super().__init__(f"ce__digit_{col}_pos{digit_pos}")
        self.col = col
        self.digit_pos = digit_pos
        self.digit_gen_ = None
        self.ce_ = None

    def fit_transform(self, df, y, fold_indices):
        self.digit_gen_ = DigitFeature(self.col, self.digit_pos)
        digit_df = self.digit_gen_.fit_transform(df, y, fold_indices)
        digit_col_name = self.digit_gen_.name

        tmp_df = df.copy()
        tmp_df[digit_col_name] = digit_df[digit_col_name]

        self.ce_ = CountEncoding(digit_col_name)
        self.ce_.name = self.name
        result = self.ce_.fit_transform(tmp_df, y, fold_indices)
        result.columns = [self.name]
        return result

    def transform(self, df):
        digit_df = self.digit_gen_.transform(df)
        digit_col_name = self.digit_gen_.name
        tmp_df = df.copy()
        tmp_df[digit_col_name] = digit_df[digit_col_name]
        result = self.ce_.transform(tmp_df)
        result.columns = [self.name]
        return result


class DigitBasePairTE(FeatureGenerator):
    """Interaction of digit of numerical col with a base categorical col, then TE. OOF — no leakage."""

    def __init__(self, num_col: str, digit_pos: int, cat_col: str, smoothing: float = 20.0):
        super().__init__(f"te__digit_{num_col}_d{digit_pos}_x_{cat_col}")
        self.num_col = num_col
        self.digit_pos = digit_pos
        self.cat_col = cat_col
        self.smoothing = smoothing
        self.digit_gen_ = None
        self.te_ = None

    def _make_pair(self, df, digit_vals):
        return digit_vals.astype(str) + "__" + df[self.cat_col].astype(str)

    def fit_transform(self, df, y, fold_indices):
        self.digit_gen_ = DigitFeature(self.num_col, self.digit_pos)
        digit_df = self.digit_gen_.fit_transform(df, y, fold_indices)
        digit_vals = digit_df.iloc[:, 0]

        tmp_df = df.copy()
        tmp_df["__dbc__"] = self._make_pair(df, digit_vals)

        self.te_ = TargetEncoding("__dbc__", self.smoothing)
        self.te_.name = self.name
        result = self.te_.fit_transform(tmp_df, y, fold_indices)
        result.columns = [self.name]
        return result

    def transform(self, df):
        digit_df = self.digit_gen_.transform(df)
        digit_vals = digit_df.iloc[:, 0]
        tmp_df = df.copy()
        tmp_df["__dbc__"] = self._make_pair(df, digit_vals)
        result = self.te_.transform(tmp_df)
        result.columns = [self.name]
        return result


# ============================================================================
# FEATURE CANDIDATE BUILDER
# ============================================================================

class FeatureCandidateBuilder:
    """
    Given a dataframe, auto-detect column types and build a list of
    FeatureGenerator candidates covering all strategies described.
    """

    def __init__(
        self,
        cat_cols: Optional[List[str]] = None,
        num_cols: Optional[List[str]] = None,
        aux_target_cols: Optional[List[str]] = None,
        max_pair_cols: int = 30,
        max_digit_positions: int = 4,
        max_digit_interaction_order: int = 3,
        rounding_decimals: List[int] = None,
        quantile_bins: List[int] = None,
    ):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.aux_target_cols = aux_target_cols or []
        self.max_pair_cols = max_pair_cols
        self.max_digit_positions = max_digit_positions
        self.max_digit_interaction_order = max_digit_interaction_order
        self.rounding_decimals = rounding_decimals or [-2, -1, 0, 1, 2]
        self.quantile_bins = quantile_bins or [5, 10, 20]

    @staticmethod
    def auto_detect_columns(df: pd.DataFrame, target_col: str = None
                            ) -> Tuple[List[str], List[str]]:
        """Auto-detect categorical and numerical columns."""
        cat_cols, num_cols = [], []
        for c in df.columns:
            if target_col and c == target_col:
                continue
            if c.lower() == "id":
                continue
            if df[c].dtype == "object" or df[c].dtype.name == "category":
                cat_cols.append(c)
            elif df[c].nunique() <= 30 and df[c].dtype in ["int64", "int32", "int16", "int8"]:
                cat_cols.append(c)
            else:
                num_cols.append(c)
        return cat_cols, num_cols

    def build(self, df: pd.DataFrame, y: pd.Series) -> List[FeatureGenerator]:
        """Return a list of all candidate feature generators, ordered by expected impact."""
        if self.cat_cols is None or self.num_cols is None:
            auto_cat, auto_num = self.auto_detect_columns(df)
            if self.cat_cols is None:
                self.cat_cols = auto_cat
            if self.num_cols is None:
                self.num_cols = auto_num

        all_cols = self.cat_cols + self.num_cols
        candidates: List[FeatureGenerator] = []

        print(f"[AutoFE] Detected {len(self.cat_cols)} cat cols, {len(self.num_cols)} num cols")
        print(f"[AutoFE] Cat: {self.cat_cols[:10]}{'...' if len(self.cat_cols) > 10 else ''}")
        print(f"[AutoFE] Num: {self.num_cols[:10]}{'...' if len(self.num_cols) > 10 else ''}")

        # Limit pairing columns
        pair_cols = all_cols[:self.max_pair_cols]

        # =====================================================================
        # ORDER: Best-performing feature types first, weakest last.
        # =====================================================================

        # --- 1) TE of base features (single columns) — typically strongest ---
        for c in all_cols:
            candidates.append(TargetEncoding(c))

        # --- 2) CE of base features (single columns) — very strong, no target needed ---
        for c in all_cols:
            candidates.append(CountEncoding(c))

        # --- 3) TE of pair interactions — powerful high-cardinality encoding ---
        for a, b in combinations(pair_cols, 2):
            candidates.append(TargetEncodingOnPair(a, b))

        # --- 4) CE of pair interactions ---
        for a, b in combinations(pair_cols, 2):
            candidates.append(CountEncodingOnPair(a, b))

        # --- 5) Frequency encoding ---
        for c in all_cols:
            candidates.append(FrequencyEncoding(c))

        # --- 6) TE with auxiliary targets ---
        for aux_col in self.aux_target_cols:
            if aux_col in df.columns:
                for c in all_cols:
                    candidates.append(TargetEncodingAuxTarget(c, aux_col))

        # --- 7) Arithmetic interactions for numericals ---
        num_arith = self.num_cols[:min(len(self.num_cols), 15)]
        for a, b in combinations(num_arith, 2):
            for op in ["add", "sub", "mul", "div"]:
                candidates.append(ArithmeticInteraction(a, b, op))

        # --- 8) Pairwise combinations of base features (label-encoded pairs) ---
        for a, b in combinations(pair_cols, 2):
            candidates.append(PairInteraction(a, b))

        # --- 9) TE/CE of digit features ---
        for c in self.num_cols[:10]:
            for d in range(min(self.max_digit_positions, 3)):
                candidates.append(TargetEncodingOnDigit(c, d))
                candidates.append(CountEncodingOnDigit(c, d))

        # --- 10) Combination of digits and base features (digit x cat -> TE) ---
        for c_num in self.num_cols[:10]:
            for c_cat in self.cat_cols[:10]:
                candidates.append(DigitBasePairTE(c_num, 0, c_cat))

        # --- 11) Quantile binning ---
        for c in self.num_cols:
            for nb in self.quantile_bins:
                candidates.append(QuantileBinFeature(c, n_bins=nb))

        # --- 12) Digit features for numerical columns ---
        for c in self.num_cols:
            for d in range(self.max_digit_positions):
                candidates.append(DigitFeature(c, d))

        # --- 13) Digit interactions WITHIN same feature (pairs, triples) ---
        for c in self.num_cols:
            digit_positions = list(range(min(self.max_digit_positions, 4)))
            for order in range(2, min(self.max_digit_interaction_order + 1, len(digit_positions) + 1)):
                for combo in combinations(digit_positions, order):
                    candidates.append(DigitInteraction([(c, d) for d in combo]))

        # --- 14) Digit interactions ACROSS features (pairs, triples) ---
        num_limited = self.num_cols[:min(len(self.num_cols), 10)]
        if len(num_limited) >= 2:
            for col_combo in combinations(num_limited, 2):
                candidates.append(DigitInteraction([(c, 0) for c in col_combo]))
                candidates.append(DigitInteraction([(c, 1) for c in col_combo]))
            if len(num_limited) >= 3:
                for col_combo in combinations(num_limited[:8], 3):
                    candidates.append(DigitInteraction([(c, 0) for c in col_combo]))
            if len(num_limited) >= 4:
                for col_combo in combinations(num_limited[:6], 4):
                    candidates.append(DigitInteraction([(c, 0) for c in col_combo]))

        # --- 15) Rounding features ---
        for c in self.num_cols:
            for dec in self.rounding_decimals:
                candidates.append(RoundFeature(c, dec))

        # --- 16) Num-to-Cat conversion ---
        for c in self.num_cols:
            candidates.append(NumToCat(c, n_bins=5))
            candidates.append(NumToCat(c, n_bins=10))

        print(f"[AutoFE] Total candidates generated: {len(candidates)}")
        return candidates


# ============================================================================
# XGB CROSS-VALIDATION ENGINE
# ============================================================================

class XGBCVEngine:
    """
    Lightweight XGBoost cross-validation engine.
    Supports custom xgb_params or uses sensible defaults.
    GPU mode uses tree_method='hist' + device='cuda'.
    """

    def __init__(self, task: str = "classification", n_folds: int = 5,
                 random_state: int = 42, use_gpu: bool = False,
                 metric_fn=None, metric_direction: str = None,
                 xgb_params: Optional[Dict[str, Any]] = None):
        self.task = task
        self.n_folds = n_folds
        self.random_state = random_state
        self.use_gpu = use_gpu
        self.xgb_params = xgb_params  # User-supplied params (or None for defaults)

        if metric_fn is not None:
            self.metric_fn = metric_fn
            self.metric_direction = metric_direction or "maximize"
        else:
            self.metric_fn, self.metric_direction = _default_metric(task)

    def get_fold_indices(self, X: pd.DataFrame, y: pd.Series
                        ) -> List[Tuple[np.ndarray, np.ndarray]]:
        if self.task == "classification":
            kf = StratifiedKFold(n_splits=self.n_folds, shuffle=True,
                                 random_state=self.random_state)
        else:
            kf = KFold(n_splits=self.n_folds, shuffle=True,
                       random_state=self.random_state)
        return list(kf.split(X, y))

    def _get_params(self) -> Dict[str, Any]:
        """
        Build XGBoost parameters.
        If user supplied xgb_params, use those as base and only fill in
        missing defaults for GPU/device settings.
        Otherwise use sensible defaults.
        """
        if self.xgb_params is not None:
            # Start from user params
            params = dict(self.xgb_params)
            # Ensure GPU settings if use_gpu and user didn't specify
            if self.use_gpu:
                params.setdefault("tree_method", "hist")
                params.setdefault("device", "cuda")
            else:
                params.setdefault("tree_method", "hist")
            # Ensure some essentials
            params.setdefault("random_state", self.random_state)
            params.setdefault("verbosity", 0)
        else:
            # Default params
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

    def evaluate(self, X: pd.DataFrame, y: pd.Series,
                 fold_indices: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """Run K-fold CV and return the mean score."""
        params = self._get_params()
        scores = []

        # Extract early_stopping_rounds from params if user set it, else default
        early_stopping_rounds = params.pop("early_stopping_rounds", 50)

        # Label encode categoricals for xgb
        X_processed = X.copy()
        label_encoders = {}
        for c in X_processed.columns:
            if X_processed[c].dtype == "object" or X_processed[c].dtype.name == "category":
                le = LabelEncoder()
                X_processed[c] = le.fit_transform(X_processed[c].fillna("__NAN__").astype(str))
                label_encoders[c] = le

        # Handle NaN in numericals
        X_processed = X_processed.fillna(-999)

        # Ensure all numeric
        for c in X_processed.columns:
            if X_processed[c].dtype not in [np.float64, np.float32, np.int64, np.int32,
                                            np.int16, np.int8, np.float16, np.uint8]:
                try:
                    X_processed[c] = X_processed[c].astype(float)
                except Exception:
                    X_processed[c] = LabelEncoder().fit_transform(X_processed[c].astype(str))

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
                X_tr, y_tr,
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

        return np.mean(scores)


# ============================================================================
# MAIN AUTOFE CLASS
# ============================================================================

class AutoFE:
    """
    Main interface for automatic feature engineering and selection.

    Usage:
        autofe = AutoFE(task="classification", time_budget=3600)
        X_train_new, X_test_new = autofe.fit_select(
            X_train, y_train, X_test,
            cat_cols=None,  # auto-detect
            num_cols=None,  # auto-detect
            aux_target_cols=["employment_status", "debt_to_income_ratio"],
        )

    Custom XGBoost parameters:
        autofe = AutoFE(
            task="classification",
            xgb_params={
                "n_estimators": 1000,
                "max_depth": 8,
                "learning_rate": 0.05,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "early_stopping_rounds": 100,
            }
        )
    """

    def __init__(
        self,
        task: str = "auto",
        n_folds: int = 5,
        time_budget: Optional[float] = None,  # seconds
        random_state: int = 42,
        metric_fn=None,
        metric_direction: str = None,
        max_pair_cols: int = 20,
        max_digit_positions: int = 4,
        max_digit_interaction_order: int = 3,
        rounding_decimals: List[int] = None,
        quantile_bins: List[int] = None,
        verbose: bool = True,
        xgb_params: Optional[Dict[str, Any]] = None,
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

        self.selected_generators_: List[FeatureGenerator] = []
        self.base_score_: float = None
        self.best_score_: float = None
        self.history_: List[Dict[str, Any]] = []

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def fit_select(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        cat_cols: Optional[List[str]] = None,
        num_cols: Optional[List[str]] = None,
        aux_target_cols: Optional[List[str]] = None,
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Fit and greedily select features.

        Returns:
            If X_test is provided: (X_train_augmented, X_test_augmented)
            Else: X_train_augmented
        """
        start_time = time.time()

        # Infer task
        if self.task == "auto":
            self.task = _infer_task(y_train)
            self._log(f"[AutoFE] Inferred task: {self.task}")

        # Encode target if needed
        self._target_le = None
        y = y_train.copy()
        if self.task == "classification" and y.dtype == "object":
            self._target_le = LabelEncoder()
            y = pd.Series(self._target_le.fit_transform(y), index=y.index)

        # GPU check
        use_gpu = _detect_gpu()
        self._log(f"[AutoFE] GPU available: {use_gpu}")

        # Build engine
        engine = XGBCVEngine(
            task=self.task, n_folds=self.n_folds,
            random_state=self.random_state, use_gpu=use_gpu,
            metric_fn=self.metric_fn, metric_direction=self.metric_direction,
            xgb_params=self.xgb_params,
        )

        # Determine metric direction if using default
        if self.metric_direction is None:
            self.metric_direction = engine.metric_direction

        # Get fold indices
        fold_indices = engine.get_fold_indices(X_train, y)

        # Build feature candidates
        builder = FeatureCandidateBuilder(
            cat_cols=cat_cols, num_cols=num_cols,
            aux_target_cols=aux_target_cols or [],
            max_pair_cols=self.max_pair_cols,
            max_digit_positions=self.max_digit_positions,
            max_digit_interaction_order=self.max_digit_interaction_order,
            rounding_decimals=self.rounding_decimals,
            quantile_bins=self.quantile_bins,
        )
        candidates = builder.build(X_train, y)

        # Prepare base feature matrix
        X_current = X_train.copy()

        # Baseline score
        self._log("[AutoFE] Computing baseline score...")
        self.base_score_ = engine.evaluate(X_current, y, fold_indices)
        self.best_score_ = self.base_score_
        self._log(f"[AutoFE] Baseline CV score: {self.base_score_:.6f}")

        # Greedy forward selection
        self.selected_generators_ = []
        total_candidates = len(candidates)

        for i, gen in enumerate(candidates):
            # Time budget check
            elapsed = time.time() - start_time
            if self.time_budget is not None and elapsed > self.time_budget:
                self._log(f"[AutoFE] Time budget exhausted ({elapsed:.0f}s). Stopping.")
                break

            eta = ""
            if i > 0 and self.time_budget:
                rate = elapsed / i
                remaining = (total_candidates - i) * rate
                eta = f" | ETA: {remaining:.0f}s"

            try:
                # Generate feature
                new_feat_df = gen.fit_transform(X_current, y, fold_indices)

                # Add to current features
                X_trial = pd.concat([X_current, new_feat_df], axis=1)

                # Evaluate
                trial_score = engine.evaluate(X_trial, y, fold_indices)

                improved = _is_improvement(trial_score, self.best_score_,
                                           self.metric_direction)

                status = "✓ KEEP" if improved else "✗ DROP"
                self._log(
                    f"[{i + 1}/{total_candidates}] {gen.name:<60s} "
                    f"score={trial_score:.6f} best={self.best_score_:.6f} "
                    f"{status}{eta}"
                )

                self.history_.append({
                    "step": i + 1,
                    "name": gen.name,
                    "score": trial_score,
                    "best_score": self.best_score_,
                    "kept": improved,
                    "elapsed": time.time() - start_time,
                })

                if improved:
                    self.best_score_ = trial_score
                    self.selected_generators_.append(gen)
                    X_current = X_trial
                else:
                    del X_trial
                    gc.collect()

            except Exception as e:
                self._log(f"[{i + 1}/{total_candidates}] {gen.name:<60s} ERROR: {str(e)[:80]}")
                self.history_.append({
                    "step": i + 1,
                    "name": gen.name,
                    "score": None,
                    "best_score": self.best_score_,
                    "kept": False,
                    "elapsed": time.time() - start_time,
                    "error": str(e),
                })

        elapsed_total = time.time() - start_time
        self._log(f"\n[AutoFE] ========== DONE ==========")
        self._log(f"[AutoFE] Baseline score : {self.base_score_:.6f}")
        self._log(f"[AutoFE] Best score     : {self.best_score_:.6f}")
        self._log(f"[AutoFE] Features added : {len(self.selected_generators_)}")
        self._log(f"[AutoFE] Total time     : {elapsed_total:.1f}s")
        self._log(f"[AutoFE] Selected features:")
        for g in self.selected_generators_:
            self._log(f"    - {g.name}")

        # Build final datasets
        # Re-fit selected generators on full training data with OOF encoding
        X_train_final = X_train.copy()
        for gen in self.selected_generators_:
            feat_df = gen.fit_transform(X_train_final, y, fold_indices)
            X_train_final = pd.concat([X_train_final, feat_df], axis=1)

        if X_test is not None:
            X_test_final = X_test.copy()
            for gen in self.selected_generators_:
                feat_df = gen.transform(X_test_final)
                X_test_final = pd.concat([X_test_final, feat_df], axis=1)
            return X_train_final, X_test_final

        return X_train_final

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply selected feature generators to new data."""
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
    metric_direction: str = None,
    random_state: int = 42,
    verbose: bool = True,
    max_pair_cols: int = 20,
    max_digit_positions: int = 4,
    xgb_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    One-liner convenience function for feature selection.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    X_test : pd.DataFrame, optional
        Test features. If provided, transformed test set is returned.
    task : str
        'classification', 'regression', or 'auto'.
    cat_cols : list of str, optional
        Categorical columns. Auto-detected if None.
    num_cols : list of str, optional
        Numerical columns. Auto-detected if None.
    aux_target_cols : list of str, optional
        Columns in X_train to use as auxiliary targets for TE.
    n_folds : int
        Number of CV folds.
    time_budget : float, optional
        Maximum time in seconds.
    metric_fn : callable, optional
        Custom metric function (y_true, y_pred) -> float.
    metric_direction : str, optional
        'maximize' or 'minimize'.
    random_state : int
        Random seed.
    verbose : bool
        Print progress.
    max_pair_cols : int
        Max columns to consider for pairwise interactions.
    max_digit_positions : int
        Max digit positions to extract.
    xgb_params : dict, optional
        Custom XGBoost parameters. If None, sensible defaults are used.
        Example:
            {
                "n_estimators": 1000,
                "max_depth": 8,
                "learning_rate": 0.05,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "reg_alpha": 0.05,
                "reg_lambda": 1.5,
                "early_stopping_rounds": 100,
            }
        GPU settings (tree_method='hist', device='cuda') are auto-applied
        when GPU is detected, but can be overridden via this dict.

    Returns
    -------
    dict with keys:
        'X_train': augmented training DataFrame
        'X_test': augmented test DataFrame (if X_test provided)
        'autofe': the fitted AutoFE object
        'history': selection history DataFrame
        'selected_features': list of selected feature names
        'base_score': baseline CV score
        'best_score': best CV score after selection
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
    )

    result = autofe.fit_select(
        X_train, y_train, X_test,
        cat_cols=cat_cols, num_cols=num_cols,
        aux_target_cols=aux_target_cols,
    )

    output = {
        "autofe": autofe,
        "history": autofe.get_history(),
        "selected_features": autofe.get_selected_feature_names(),
        "base_score": autofe.base_score_,
        "best_score": autofe.best_score_,
    }

    if X_test is not None:
        output["X_train"] = result[0]
        output["X_test"] = result[1]
    else:
        output["X_train"] = result

    return output


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
