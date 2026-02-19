"""
Feature generators for AutoFE-PG.

Each generator produces one or more columns and implements:
- fit_transform(df, y, fold_indices) → pd.DataFrame (train features)
- transform(df) → pd.DataFrame (test features)

All generators that use the target y do so in an out-of-fold manner
to prevent target leakage.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from sklearn.preprocessing import LabelEncoder


# ============================================================================
# BASE CLASS
# ============================================================================
class FeatureGenerator:
    """Abstract base class for feature generators.

    Parameters
    ----------
    name : str
        Human-readable name for the generated feature(s).
    """

    def __init__(self, name: str):
        self.name = name

    def fit_transform(
        self,
        df: pd.DataFrame,
        y: pd.Series,
        fold_indices: List[Tuple[np.ndarray, np.ndarray]],
    ) -> pd.DataFrame:
        """Generate features for the training set.

        Parameters
        ----------
        df : pd.DataFrame
            Training features.
        y : pd.Series
            Training target.
        fold_indices : list of (train_idx, val_idx) tuples
            Cross-validation fold indices.

        Returns
        -------
        pd.DataFrame
            DataFrame with generated feature column(s).
        """
        raise NotImplementedError

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate features for new data (test set).

        Parameters
        ----------
        df : pd.DataFrame
            New data features.

        Returns
        -------
        pd.DataFrame
            DataFrame with generated feature column(s).
        """
        raise NotImplementedError


# ============================================================================
# PAIR INTERACTION
# ============================================================================
class PairInteraction(FeatureGenerator):
    """Create a label-encoded interaction feature between two columns.

    Parameters
    ----------
    col_a : str
        First column name.
    col_b : str
        Second column name.
    """

    def __init__(self, col_a: str, col_b: str):
        super().__init__(f"pair__{col_a}_x_{col_b}")
        self.col_a = col_a
        self.col_b = col_b
        self._le_dict = None

    def fit_transform(self, df, y, fold_indices):
        s = df[self.col_a].astype(str) + "__" + df[self.col_b].astype(str)
        le = LabelEncoder()
        vals = le.fit_transform(s.fillna("__NAN__"))
        self._le_dict = {cls: idx for idx, cls in enumerate(le.classes_)}
        return pd.DataFrame({self.name: vals}, index=df.index)

    def transform(self, df):
        s = df[self.col_a].astype(str) + "__" + df[self.col_b].astype(str)
        s = s.fillna("__NAN__")
        mapped = s.map(self._le_dict).fillna(-1).astype(int)
        return pd.DataFrame({self.name: mapped.values}, index=df.index)


# ============================================================================
# TARGET ENCODING (OUT-OF-FOLD)
# ============================================================================
class TargetEncoding(FeatureGenerator):
    """Out-of-fold target encoding for a single column.

    Uses smoothing to regularize estimates for rare categories.

    Parameters
    ----------
    col : str
        Column to encode.
    smoothing : float
        Smoothing factor (higher = more regularization toward global mean).
    suffix : str
        Optional suffix for the feature name.
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
            smooth = (
                stats["count"] * stats["mean"] + self.smoothing * self.global_mean_
            ) / (stats["count"] + self.smoothing)
            result.iloc[va_idx] = df[self.col].iloc[va_idx].map(smooth)

        full_stats = y.groupby(df[self.col]).agg(["mean", "count"])
        self.mapping_ = (
            full_stats["count"] * full_stats["mean"]
            + self.smoothing * self.global_mean_
        ) / (full_stats["count"] + self.smoothing)

        result = result.fillna(self.global_mean_)
        return pd.DataFrame({self.name: result.values}, index=df.index)

    def transform(self, df):
        vals = df[self.col].map(self.mapping_).fillna(self.global_mean_)
        return pd.DataFrame({self.name: vals.values}, index=df.index)


# ============================================================================
# COUNT ENCODING
# ============================================================================
class CountEncoding(FeatureGenerator):
    """Count encoding for a single column.

    Parameters
    ----------
    col : str
        Column to encode.
    suffix : str
        Optional suffix for the feature name.
    """

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


# ============================================================================
# DIGIT EXTRACTION
# ============================================================================
class DigitFeature(FeatureGenerator):
    """Extract the i-th digit of a numerical feature using vectorized modular arithmetic.

    Parameters
    ----------
    col : str
        Numerical column.
    digit_pos : int
        Digit position (0 = ones, 1 = tens, etc.).
    """

    def __init__(self, col: str, digit_pos: int):
        super().__init__(f"digit__{col}_pos{digit_pos}")
        self.col = col
        self.digit_pos = digit_pos

    def _extract(self, series: pd.Series) -> pd.Series:
        abs_int = series.fillna(0).abs().astype(np.int64)
        divisor = np.int64(10 ** self.digit_pos)
        return ((abs_int // divisor) % 10).astype(np.int8)

    def fit_transform(self, df, y, fold_indices):
        vals = self._extract(df[self.col])
        return pd.DataFrame({self.name: vals.values}, index=df.index)

    def transform(self, df):
        vals = self._extract(df[self.col])
        return pd.DataFrame({self.name: vals.values}, index=df.index)


# ============================================================================
# DIGIT INTERACTION
# ============================================================================
class DigitInteraction(FeatureGenerator):
    """Interaction between digits of numerical features using vectorized ops.

    Parameters
    ----------
    col_digit_pairs : list of (str, int) tuples
        Each tuple is ``(column_name, digit_position)``.
    """

    def __init__(self, col_digit_pairs: List[Tuple[str, int]]):
        name_parts = "_".join([f"{c}d{d}" for c, d in col_digit_pairs])
        super().__init__(f"digitint__{name_parts}")
        self.col_digit_pairs = col_digit_pairs

    def _extract_digit(self, series: pd.Series, pos: int) -> np.ndarray:
        abs_int = series.fillna(0).abs().astype(np.int64).values
        divisor = np.int64(10 ** pos)
        return (abs_int // divisor) % 10

    def _compute(self, df: pd.DataFrame) -> np.ndarray:
        result = np.zeros(len(df), dtype=np.int64)
        for col, dpos in self.col_digit_pairs:
            digits = self._extract_digit(df[col], dpos)
            result = result * 10 + digits
        return result

    def fit_transform(self, df, y, fold_indices):
        vals = self._compute(df)
        return pd.DataFrame({self.name: vals}, index=df.index)

    def transform(self, df):
        vals = self._compute(df)
        return pd.DataFrame({self.name: vals}, index=df.index)


# ============================================================================
# ROUNDING
# ============================================================================
class RoundFeature(FeatureGenerator):
    """Round a numerical feature to a given number of decimals or magnitude.

    Parameters
    ----------
    col : str
        Numerical column.
    decimals : int
        Number of decimal places (negative values round to tens, hundreds, etc.).
    """

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


# ============================================================================
# QUANTILE BINNING
# ============================================================================
class QuantileBinFeature(FeatureGenerator):
    """Bin a numerical column into quantile bins.

    Parameters
    ----------
    col : str
        Numerical column.
    n_bins : int
        Number of quantile bins.
    """

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
            binned, edges = pd.qcut(
                vals, q=self.n_bins, labels=False, retbins=True, duplicates="drop"
            )
        except Exception:
            binned = pd.Series(0, index=df.index)
            edges = None
        self.bin_edges_ = edges
        return pd.DataFrame({self.name: binned.values}, index=df.index)

    def transform(self, df):
        vals = df[self.col].fillna(
            self.fill_median_ if self.fill_median_ is not None else 0
        )
        if self.bin_edges_ is not None:
            binned = pd.cut(vals, bins=self.bin_edges_, labels=False, include_lowest=True)
            binned = binned.fillna(0).astype(int)
        else:
            binned = pd.Series(0, index=df.index)
        return pd.DataFrame({self.name: binned.values}, index=df.index)


# ============================================================================
# NUM TO CAT
# ============================================================================
class NumToCat(FeatureGenerator):
    """Convert a numerical column into a categorical one via equal-width binning.

    Parameters
    ----------
    col : str
        Numerical column.
    n_bins : int
        Number of equal-width bins.
    """

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
            binned, edges = pd.cut(
                vals, bins=self.n_bins, labels=False, retbins=True, duplicates="drop"
            )
        except Exception:
            binned = pd.Series(0, index=df.index)
            edges = None
        self.bin_edges_ = edges
        return pd.DataFrame(
            {self.name: binned.fillna(0).astype(int).values}, index=df.index
        )

    def transform(self, df):
        vals = df[self.col].fillna(
            self.fill_median_ if self.fill_median_ is not None else 0
        )
        if self.bin_edges_ is not None:
            binned = pd.cut(vals, bins=self.bin_edges_, labels=False, include_lowest=True)
            binned = binned.fillna(0).astype(int)
        else:
            binned = pd.Series(0, index=df.index)
        return pd.DataFrame({self.name: binned.values}, index=df.index)


# ============================================================================
# TARGET ENCODING WITH AUXILIARY TARGET
# ============================================================================
class TargetEncodingAuxTarget(FeatureGenerator):
    """Target-encode a column using a different column as the 'target'.

    Out-of-fold to prevent leakage even with auxiliary targets.

    Parameters
    ----------
    col : str
        Column to encode.
    aux_target_col : str
        Column in the DataFrame to use as the encoding target.
    smoothing : float
        Smoothing factor.
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
        if aux.dtype == "object" or aux.dtype.name == "category":
            self._aux_le = LabelEncoder()
            aux = pd.Series(
                self._aux_le.fit_transform(aux.fillna("__NAN__").astype(str)),
                index=df.index,
                dtype=float,
            )
        else:
            aux = aux.fillna(aux.mean())

        self.global_mean_ = aux.mean()
        result = pd.Series(np.nan, index=df.index, dtype=float)

        for tr_idx, va_idx in fold_indices:
            tr_col = df[self.col].iloc[tr_idx]
            tr_aux = aux.iloc[tr_idx]
            stats = tr_aux.groupby(tr_col).agg(["mean", "count"])
            smooth = (
                stats["count"] * stats["mean"] + self.smoothing * self.global_mean_
            ) / (stats["count"] + self.smoothing)
            result.iloc[va_idx] = df[self.col].iloc[va_idx].map(smooth)

        full_stats = aux.groupby(df[self.col]).agg(["mean", "count"])
        self.mapping_ = (
            full_stats["count"] * full_stats["mean"]
            + self.smoothing * self.global_mean_
        ) / (full_stats["count"] + self.smoothing)

        result = result.fillna(self.global_mean_)
        return pd.DataFrame({self.name: result.values}, index=df.index)

    def transform(self, df):
        if self.mapping_ is None:
            return pd.DataFrame({self.name: np.zeros(len(df))}, index=df.index)
        vals = df[self.col].map(self.mapping_).fillna(self.global_mean_)
        return pd.DataFrame({self.name: vals.values}, index=df.index)


# ============================================================================
# ARITHMETIC INTERACTIONS
# ============================================================================
class ArithmeticInteraction(FeatureGenerator):
    """Sum, difference, product, or ratio between two numerical columns.

    Parameters
    ----------
    col_a : str
        First numerical column.
    col_b : str
        Second numerical column.
    op : str
        Operation: ``'add'``, ``'sub'``, ``'mul'``, or ``'div'``.
    """

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
        return a + b

    def fit_transform(self, df, y, fold_indices):
        vals = self._compute(df)
        return pd.DataFrame({self.name: vals.values}, index=df.index)

    def transform(self, df):
        vals = self._compute(df)
        return pd.DataFrame({self.name: vals.values}, index=df.index)


# ============================================================================
# FREQUENCY ENCODING
# ============================================================================
class FrequencyEncoding(FeatureGenerator):
    """Frequency (normalized count) encoding for a column.

    Parameters
    ----------
    col : str
        Column to encode.
    """

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
# MISSING INDICATOR
# ============================================================================
class MissingIndicator(FeatureGenerator):
    """Binary flag for whether a value was originally NaN.

    Parameters
    ----------
    col : str
        Column to check for missing values.
    """

    def __init__(self, col: str):
        super().__init__(f"missing__{col}")
        self.col = col

    def fit_transform(self, df, y, fold_indices):
        vals = df[self.col].isna().astype(np.int8)
        return pd.DataFrame({self.name: vals.values}, index=df.index)

    def transform(self, df):
        vals = df[self.col].isna().astype(np.int8)
        return pd.DataFrame({self.name: vals.values}, index=df.index)


# ============================================================================
# GROUP STATISTICS
# ============================================================================
class GroupStatFeature(FeatureGenerator):
    """Group-level statistics for a numerical column grouped by a categorical column.

    Parameters
    ----------
    num_col : str
        Numerical column to aggregate.
    cat_col : str
        Categorical column to group by.
    stat : str
        Statistic: ``'mean'``, ``'std'``, ``'min'``, ``'max'``, or ``'median'``.
    """

    def __init__(self, num_col: str, cat_col: str, stat: str = "mean"):
        super().__init__(f"grpstat__{num_col}_by_{cat_col}_{stat}")
        self.num_col = num_col
        self.cat_col = cat_col
        self.stat = stat
        self.mapping_ = None
        self.fill_value_ = None

    def fit_transform(self, df, y, fold_indices):
        grouped = df.groupby(self.cat_col)[self.num_col]
        if self.stat == "mean":
            self.mapping_ = grouped.mean()
        elif self.stat == "std":
            self.mapping_ = grouped.std().fillna(0)
        elif self.stat == "min":
            self.mapping_ = grouped.min()
        elif self.stat == "max":
            self.mapping_ = grouped.max()
        elif self.stat == "median":
            self.mapping_ = grouped.median()
        else:
            self.mapping_ = grouped.mean()

        self.fill_value_ = (
            df[self.num_col].agg(self.stat) if self.stat != "std" else 0
        )
        if pd.isna(self.fill_value_):
            self.fill_value_ = 0

        vals = df[self.cat_col].map(self.mapping_).fillna(self.fill_value_)
        return pd.DataFrame({self.name: vals.values}, index=df.index)

    def transform(self, df):
        vals = df[self.cat_col].map(self.mapping_).fillna(self.fill_value_)
        return pd.DataFrame({self.name: vals.values}, index=df.index)


class GroupDeviationFeature(FeatureGenerator):
    """Value minus group mean (or ratio to group mean) for a numerical column.

    Parameters
    ----------
    num_col : str
        Numerical column.
    cat_col : str
        Categorical column to group by.
    mode : str
        ``'diff'`` for subtraction or ``'ratio'`` for division.
    """

    def __init__(self, num_col: str, cat_col: str, mode: str = "diff"):
        super().__init__(f"grpdev__{num_col}_by_{cat_col}_{mode}")
        self.num_col = num_col
        self.cat_col = cat_col
        self.mode = mode
        self.group_mean_ = None
        self.global_mean_ = None

    def fit_transform(self, df, y, fold_indices):
        self.group_mean_ = df.groupby(self.cat_col)[self.num_col].mean()
        self.global_mean_ = df[self.num_col].mean()
        if pd.isna(self.global_mean_):
            self.global_mean_ = 0

        grp_vals = df[self.cat_col].map(self.group_mean_).fillna(self.global_mean_)
        num_vals = df[self.num_col].fillna(self.global_mean_)

        if self.mode == "diff":
            result = num_vals - grp_vals
        else:
            result = num_vals / (grp_vals + 1e-8)

        return pd.DataFrame({self.name: result.values}, index=df.index)

    def transform(self, df):
        grp_vals = df[self.cat_col].map(self.group_mean_).fillna(self.global_mean_)
        num_vals = df[self.num_col].fillna(self.global_mean_)

        if self.mode == "diff":
            result = num_vals - grp_vals
        else:
            result = num_vals / (grp_vals + 1e-8)

        return pd.DataFrame({self.name: result.values}, index=df.index)


# ============================================================================
# UNARY TRANSFORMS
# ============================================================================
class UnaryTransform(FeatureGenerator):
    """Unary nonlinear transformations: log1p, sqrt, square, reciprocal.

    Parameters
    ----------
    col : str
        Numerical column.
    transform_type : str
        One of ``'log1p'``, ``'sqrt'``, ``'square'``, ``'reciprocal'``.
    """

    def __init__(self, col: str, transform_type: str = "log1p"):
        super().__init__(f"unary__{col}_{transform_type}")
        self.col = col
        self.transform_type = transform_type

    def _apply(self, series: pd.Series) -> pd.Series:
        vals = series.fillna(0).astype(float)
        if self.transform_type == "log1p":
            return np.log1p(np.abs(vals)) * np.sign(vals)
        elif self.transform_type == "sqrt":
            return np.sqrt(np.abs(vals)) * np.sign(vals)
        elif self.transform_type == "square":
            return vals ** 2
        elif self.transform_type == "reciprocal":
            return 1.0 / (vals + 1e-8)
        return vals

    def fit_transform(self, df, y, fold_indices):
        vals = self._apply(df[self.col])
        return pd.DataFrame({self.name: vals.values}, index=df.index)

    def transform(self, df):
        vals = self._apply(df[self.col])
        return pd.DataFrame({self.name: vals.values}, index=df.index)


# ============================================================================
# POLYNOMIAL FEATURES
# ============================================================================
class PolynomialFeature(FeatureGenerator):
    """Second-degree polynomial features: square or cross product.

    Parameters
    ----------
    col_a : str
        First numerical column.
    col_b : str, optional
        Second column (required for ``'cross'`` type).
    poly_type : str
        ``'square'`` for a² or ``'cross'`` for a×b.
    """

    def __init__(
        self,
        col_a: str,
        col_b: Optional[str] = None,
        poly_type: str = "square",
    ):
        if col_b is None:
            super().__init__(f"poly__{col_a}_{poly_type}")
        else:
            super().__init__(f"poly__{col_a}_{poly_type}_{col_b}")
        self.col_a = col_a
        self.col_b = col_b
        self.poly_type = poly_type

    def _compute(self, df):
        a = df[self.col_a].fillna(0).astype(float)
        if self.poly_type == "square":
            return a ** 2
        elif self.poly_type == "cross" and self.col_b is not None:
            b = df[self.col_b].fillna(0).astype(float)
            return a * b
        return a

    def fit_transform(self, df, y, fold_indices):
        vals = self._compute(df)
        return pd.DataFrame({self.name: vals.values}, index=df.index)

    def transform(self, df):
        vals = self._compute(df)
        return pd.DataFrame({self.name: vals.values}, index=df.index)


# ============================================================================
# COMPOSITE: TE/CE ON PAIRS
# ============================================================================
class TargetEncodingOnPair(FeatureGenerator):
    """Target encoding on a pair interaction. OOF — no leakage.

    Parameters
    ----------
    col_a : str
        First column.
    col_b : str
        Second column.
    smoothing : float
        Smoothing factor.
    """

    def __init__(self, col_a: str, col_b: str, smoothing: float = 20.0):
        super().__init__(f"te__pair_{col_a}_x_{col_b}")
        self.col_a = col_a
        self.col_b = col_b
        self.smoothing = smoothing
        self.te_ = None

    def _make_pair_col(self, df):
        return df[self.col_a].astype(str) + "__" + df[self.col_b].astype(str)

    def fit_transform(self, df, y, fold_indices):
        pair_col_name = "__pair__"
        df[pair_col_name] = self._make_pair_col(df)
        self.te_ = TargetEncoding(
            pair_col_name, self.smoothing, suffix=f"pair_{self.col_a}_{self.col_b}"
        )
        self.te_.name = self.name
        result = self.te_.fit_transform(df, y, fold_indices)
        del df[pair_col_name]
        result.columns = [self.name]
        return result

    def transform(self, df):
        pair_col_name = "__pair__"
        df[pair_col_name] = self._make_pair_col(df)
        result = self.te_.transform(df)
        del df[pair_col_name]
        result.columns = [self.name]
        return result


class CountEncodingOnPair(FeatureGenerator):
    """Count encoding on a pair interaction. No target used — no leakage.

    Parameters
    ----------
    col_a : str
        First column.
    col_b : str
        Second column.
    """

    def __init__(self, col_a: str, col_b: str):
        super().__init__(f"ce__pair_{col_a}_x_{col_b}")
        self.col_a = col_a
        self.col_b = col_b
        self.ce_ = None

    def _make_pair_col(self, df):
        return df[self.col_a].astype(str) + "__" + df[self.col_b].astype(str)

    def fit_transform(self, df, y, fold_indices):
        pair_col_name = "__pair__"
        df[pair_col_name] = self._make_pair_col(df)
        self.ce_ = CountEncoding(
            pair_col_name, suffix=f"pair_{self.col_a}_{self.col_b}"
        )
        self.ce_.name = self.name
        result = self.ce_.fit_transform(df, y, fold_indices)
        del df[pair_col_name]
        result.columns = [self.name]
        return result

    def transform(self, df):
        pair_col_name = "__pair__"
        df[pair_col_name] = self._make_pair_col(df)
        result = self.ce_.transform(df)
        del df[pair_col_name]
        result.columns = [self.name]
        return result


# ============================================================================
# COMPOSITE: TE/CE ON DIGITS
# ============================================================================
class TargetEncodingOnDigit(FeatureGenerator):
    """Target encoding on a digit feature. OOF — no leakage.

    Parameters
    ----------
    col : str
        Numerical column.
    digit_pos : int
        Digit position to extract.
    smoothing : float
        Smoothing factor.
    """

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

        df[digit_col_name] = digit_df[digit_col_name].values
        self.te_ = TargetEncoding(digit_col_name, self.smoothing)
        self.te_.name = self.name
        result = self.te_.fit_transform(df, y, fold_indices)
        del df[digit_col_name]
        result.columns = [self.name]
        return result

    def transform(self, df):
        digit_df = self.digit_gen_.transform(df)
        digit_col_name = self.digit_gen_.name
        df[digit_col_name] = digit_df[digit_col_name].values
        result = self.te_.transform(df)
        del df[digit_col_name]
        result.columns = [self.name]
        return result


class CountEncodingOnDigit(FeatureGenerator):
    """Count encoding on a digit feature. No target used — no leakage.

    Parameters
    ----------
    col : str
        Numerical column.
    digit_pos : int
        Digit position to extract.
    """

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

        df[digit_col_name] = digit_df[digit_col_name].values
        self.ce_ = CountEncoding(digit_col_name)
        self.ce_.name = self.name
        result = self.ce_.fit_transform(df, y, fold_indices)
        del df[digit_col_name]
        result.columns = [self.name]
        return result

    def transform(self, df):
        digit_df = self.digit_gen_.transform(df)
        digit_col_name = self.digit_gen_.name
        df[digit_col_name] = digit_df[digit_col_name].values
        result = self.ce_.transform(df)
        del df[digit_col_name]
        result.columns = [self.name]
        return result


# ============================================================================
# COMPOSITE: DIGIT x CATEGORY TE
# ============================================================================
class DigitBasePairTE(FeatureGenerator):
    """Interaction of a digit of a numerical column with a categorical column, then TE.

    OOF — no leakage.

    Parameters
    ----------
    num_col : str
        Numerical column.
    digit_pos : int
        Digit position to extract.
    cat_col : str
        Categorical column.
    smoothing : float
        Smoothing factor.
    """

    def __init__(
        self,
        num_col: str,
        digit_pos: int,
        cat_col: str,
        smoothing: float = 20.0,
    ):
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

        tmp_col_name = "__dbc__"
        df[tmp_col_name] = self._make_pair(df, digit_vals)
        self.te_ = TargetEncoding(tmp_col_name, self.smoothing)
        self.te_.name = self.name
        result = self.te_.fit_transform(df, y, fold_indices)
        del df[tmp_col_name]
        result.columns = [self.name]
        return result

    def transform(self, df):
        digit_df = self.digit_gen_.transform(df)
        digit_vals = digit_df.iloc[:, 0]
        tmp_col_name = "__dbc__"
        df[tmp_col_name] = self._make_pair(df, digit_vals)
        result = self.te_.transform(df)
        del df[tmp_col_name]
        result.columns = [self.name]
        return result


# ============================================================================
# DOMAIN ALIGNMENT (DE-NOISING) — Snap synthetic values to nearest real value
# ============================================================================
class DomainAlignmentFeature(FeatureGenerator):
    """Snap each value to its nearest neighbor in a reference (original) dataset.

    Forces synthetic "fuzzy" continuous values back onto the real clinical grid.
    The snapped value replaces the original for downstream modelling and the
    residual (distance to nearest real value) is exposed as an extra signal.

    Parameters
    ----------
    col : str
        Numerical column to align.
    reference_values : array-like
        Sorted unique values from the original (real) dataset for this column.
    include_residual : bool
        If True, also emit a ``_residual`` column (signed distance to snap point).
    """

    def __init__(
        self,
        col: str,
        reference_values: np.ndarray,
        include_residual: bool = True,
    ):
        super().__init__(f"align__{col}")
        self.col = col
        self.reference_values = np.sort(np.asarray(reference_values, dtype=float))
        self.include_residual = include_residual
        self.fill_median_: Optional[float] = None

    def _snap(self, series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Return (snapped_values, residuals)."""
        vals = series.fillna(self.fill_median_ if self.fill_median_ is not None else 0).values.astype(float)
        idx = np.searchsorted(self.reference_values, vals, side="left")
        idx = np.clip(idx, 1, len(self.reference_values) - 1)

        left = self.reference_values[idx - 1]
        right = self.reference_values[idx]

        snapped = np.where(np.abs(vals - left) <= np.abs(vals - right), left, right)
        residuals = vals - snapped
        return snapped, residuals

    def fit_transform(self, df, y, fold_indices):
        self.fill_median_ = df[self.col].median()
        snapped, residuals = self._snap(df[self.col])
        out = {self.name: snapped}
        if self.include_residual:
            out[f"{self.name}_residual"] = residuals
        return pd.DataFrame(out, index=df.index)

    def transform(self, df):
        snapped, residuals = self._snap(df[self.col])
        out = {self.name: snapped}
        if self.include_residual:
            out[f"{self.name}_residual"] = residuals
        return pd.DataFrame(out, index=df.index)


# ============================================================================
# BAYESIAN-STYLE PRIORS (EXTERNAL MAPPING)
# ============================================================================
class BayesianPriorFeature(FeatureGenerator):
    """Inject an external prior probability of the target for each value.

    Given a pre-computed mapping ``{value: P(target | value)}`` derived from an
    external (original) dataset, each row receives its historical probability
    as a feature.  This gives the model a "hint" without using the training
    target — zero leakage.

    Parameters
    ----------
    col : str
        Column whose values are looked up.
    prior_map : dict
        Mapping ``{value: prior_probability}``.
    global_prior : float, optional
        Fallback value for unseen keys. If None, uses the mean of the map.
    """

    def __init__(
        self,
        col: str,
        prior_map: Dict,
        global_prior: Optional[float] = None,
    ):
        super().__init__(f"prior__{col}")
        self.col = col
        self.prior_map = prior_map
        self.global_prior = (
            global_prior
            if global_prior is not None
            else float(np.mean(list(prior_map.values()))) if prior_map else 0.0
        )

    def fit_transform(self, df, y, fold_indices):
        vals = df[self.col].map(self.prior_map).fillna(self.global_prior)
        return pd.DataFrame({self.name: vals.values.astype(float)}, index=df.index)

    def transform(self, df):
        vals = df[self.col].map(self.prior_map).fillna(self.global_prior)
        return pd.DataFrame({self.name: vals.values.astype(float)}, index=df.index)


# ============================================================================
# DIMENSIONALITY EXPANSION (DUAL REPRESENTATION)
# ============================================================================
class DualRepresentationFeature(FeatureGenerator):
    """Emit both a continuous and a categorical (label-encoded) copy of a column.

    The continuous copy preserves linear / threshold trends while the
    categorical copy lets tree models create very specific, non-linear splits
    on exact values.

    Parameters
    ----------
    col : str
        Column to duplicate.
    """

    def __init__(self, col: str):
        super().__init__(f"dual__{col}")
        self.col = col
        self._le_dict: Optional[Dict] = None

    def fit_transform(self, df, y, fold_indices):
        continuous = df[self.col].fillna(0).astype(float)
        cat_key = df[self.col].fillna("__NAN__").astype(str)
        le = LabelEncoder()
        categorical = le.fit_transform(cat_key)
        self._le_dict = {cls: idx for idx, cls in enumerate(le.classes_)}
        return pd.DataFrame(
            {
                f"{self.name}_cont": continuous.values,
                f"{self.name}_cat": categorical,
            },
            index=df.index,
        )

    def transform(self, df):
        continuous = df[self.col].fillna(0).astype(float)
        cat_key = df[self.col].fillna("__NAN__").astype(str)
        categorical = cat_key.map(self._le_dict).fillna(-1).astype(int)
        return pd.DataFrame(
            {
                f"{self.name}_cont": continuous.values,
                f"{self.name}_cat": categorical.values,
            },
            index=df.index,
        )


# ============================================================================
# CROSS-DATASET FREQUENCY / DENSITY ANALYSIS
# ============================================================================
class CrossDatasetFrequencyFeature(FeatureGenerator):
    """Measure how common a value is across train, test, and original datasets.

    Computes per-value frequency across the combined ecosystem and flags
    outliers / over-represented synthetic modes.

    Parameters
    ----------
    col : str
        Column to analyse.
    ecosystem_counts : pd.Series
        Pre-computed ``value_counts()`` from the combined
        (train + test + original) data for this column.
    ecosystem_total : int
        Total row count of the combined ecosystem.
    """

    def __init__(
        self,
        col: str,
        ecosystem_counts: pd.Series,
        ecosystem_total: int,
    ):
        super().__init__(f"xfreq__{col}")
        self.col = col
        self.ecosystem_freq = (ecosystem_counts / ecosystem_total).astype(float)
        self.ecosystem_total = ecosystem_total

    def fit_transform(self, df, y, fold_indices):
        freq_vals = df[self.col].map(self.ecosystem_freq).fillna(0.0)
        return pd.DataFrame({self.name: freq_vals.values}, index=df.index)

    def transform(self, df):
        freq_vals = df[self.col].map(self.ecosystem_freq).fillna(0.0)
        return pd.DataFrame({self.name: freq_vals.values}, index=df.index)


class ValueRarityFeature(FeatureGenerator):
    """Flag how rare a value is across the data ecosystem.

    Produces a log-inverse-frequency score (higher = rarer). Useful for
    synthetic datasets where certain modes are over-represented.

    Parameters
    ----------
    col : str
        Column to analyse.
    ecosystem_counts : pd.Series
        Pre-computed ``value_counts()`` from the combined data.
    ecosystem_total : int
        Total row count of the combined ecosystem.
    """

    def __init__(
        self,
        col: str,
        ecosystem_counts: pd.Series,
        ecosystem_total: int,
    ):
        super().__init__(f"rarity__{col}")
        self.col = col
        self.ecosystem_freq = (ecosystem_counts / ecosystem_total).astype(float)
        self.max_rarity = float(np.log1p(ecosystem_total))

    def _compute(self, series: pd.Series) -> np.ndarray:
        freq = series.map(self.ecosystem_freq).fillna(0.0).values.astype(float)
        # log-inverse-frequency; clip to avoid log(0)
        rarity = np.where(
            freq > 0,
            np.log1p(1.0 / freq),
            self.max_rarity,
        )
        return rarity

    def fit_transform(self, df, y, fold_indices):
        vals = self._compute(df[self.col])
        return pd.DataFrame({self.name: vals}, index=df.index)

    def transform(self, df):
        vals = self._compute(df[self.col])
        return pd.DataFrame({self.name: vals}, index=df.index)


# ============================================================================
# EXTERNAL DATASET TARGET STATISTICS
# ============================================================================
class ExternalTargetStatFeature(FeatureGenerator):
    """Look up pre-computed target distribution statistics from an external dataset.

    Given a mapping from each value of a column to a dictionary of statistics
    (e.g. median, std, skew, count) derived from the original (real) dataset's
    target variable, this generator emits those statistics as separate numeric
    columns for each row.

    This provides the model with a calibrated "prior shape" — not just the mean
    probability but the uncertainty, skewness, and support of the target within
    each group, all derived from an external source with zero leakage.

    Parameters
    ----------
    col : str
        Column whose values are used for the lookup.
    stat_map : dict
        Mapping ``{value: {"median": float, "std": float, "skew": float,
        "count": int, ...}}``. Each value maps to a sub-dict of
        statistic-name → float.
    global_fallback : dict, optional
        Fallback statistics for unseen values. If None, defaults are computed
        from the stat_map: median of medians, 0 for std/skew, minimum count.
    """

    def __init__(
        self,
        col: str,
        stat_map: Dict,
        global_fallback: Optional[Dict[str, float]] = None,
    ):
        super().__init__(f"extstat__{col}")
        self.col = col
        self.stat_map = stat_map

        # Determine the set of statistics from the first entry
        if stat_map:
            first_key = next(iter(stat_map))
            self.stat_names = sorted(stat_map[first_key].keys())
        else:
            self.stat_names = []

        # Build global fallback
        if global_fallback is not None:
            self.global_fallback = global_fallback
        else:
            self.global_fallback = self._compute_default_fallback()

    def _compute_default_fallback(self) -> Dict[str, float]:
        """Derive sensible default fallback values from the stat_map."""
        if not self.stat_map or not self.stat_names:
            return {}

        fallback = {}
        for stat in self.stat_names:
            values = [
                v[stat] for v in self.stat_map.values()
                if stat in v and v[stat] is not None and not np.isnan(v[stat])
            ]
            if not values:
                fallback[stat] = 0.0
                continue

            if stat in ("median", "mean"):
                fallback[stat] = float(np.median(values))
            elif stat in ("std", "skew"):
                fallback[stat] = 0.0
            elif stat == "count":
                fallback[stat] = float(min(values))
            else:
                fallback[stat] = float(np.median(values))

        return fallback

    def _lookup(self, df: pd.DataFrame) -> pd.DataFrame:
        """Look up statistics for each row."""
        out = {}
        for stat in self.stat_names:
            col_name = f"extstat__{self.col}__{stat}"
            fb = self.global_fallback.get(stat, 0.0)
            mapping = {k: v.get(stat, fb) for k, v in self.stat_map.items()}
            out[col_name] = df[self.col].map(mapping).fillna(fb).astype(float)
        return pd.DataFrame(out, index=df.index)

    def fit_transform(self, df, y, fold_indices):
        return self._lookup(df)

    def transform(self, df):
        return self._lookup(df)


# ============================================================================
# OOF TARGET AGGREGATION WITH MULTIPLE STATISTICS
# ============================================================================
class OOFTargetAggFeature(FeatureGenerator):
    """Out-of-fold target aggregation with multiple statistics per group.

    Like target encoding, but emits not just the smoothed mean but also
    additional statistics of the raw target values per group: standard
    deviation, count, min, and max. All statistics are computed out-of-fold
    to prevent target leakage.

    The mean statistic uses smoothing toward the global mean for
    regularization. Other statistics (std, count, min, max) are computed
    directly from the fold's training portion without smoothing.

    Parameters
    ----------
    col : str
        Column to group by.
    stats : list of str, optional
        Statistics to compute. Default: ``["mean", "std", "count"]``.
        Supported: ``"mean"``, ``"std"``, ``"count"``, ``"min"``, ``"max"``.
    smoothing : float
        Smoothing factor applied to the mean statistic only.
    """

    def __init__(
        self,
        col: str,
        stats: Optional[List[str]] = None,
        smoothing: float = 20.0,
    ):
        super().__init__(f"oof_agg__{col}")
        self.col = col
        self.stats = stats or ["mean", "std", "count"]
        self.smoothing = smoothing
        self.global_mean_: Optional[float] = None
        self.global_std_: Optional[float] = None
        self.global_count_: Optional[float] = None
        self.global_min_: Optional[float] = None
        self.global_max_: Optional[float] = None
        self.mappings_: Optional[Dict[str, pd.Series]] = None

    def _col_name(self, stat: str) -> str:
        return f"oof_agg__{self.col}__{stat}"

    def _compute_fold_stats(
        self, group_col: pd.Series, target: pd.Series
    ) -> Dict[str, pd.Series]:
        """Compute per-group statistics from a single fold's training data."""
        grouped = target.groupby(group_col)
        result = {}

        if "mean" in self.stats:
            agg = grouped.agg(["mean", "count"])
            smoothed = (
                agg["count"] * agg["mean"] + self.smoothing * self.global_mean_
            ) / (agg["count"] + self.smoothing)
            result["mean"] = smoothed

        if "std" in self.stats:
            result["std"] = grouped.std().fillna(0.0)

        if "count" in self.stats:
            result["count"] = grouped.count().astype(float)

        if "min" in self.stats:
            result["min"] = grouped.min()

        if "max" in self.stats:
            result["max"] = grouped.max()

        return result

    def _get_fill_value(self, stat: str) -> float:
        """Return the neutral fill value for a given statistic."""
        if stat == "mean":
            return self.global_mean_ if self.global_mean_ is not None else 0.5
        elif stat == "std":
            return 0.0
        elif stat == "count":
            return 1.0
        elif stat == "min":
            return self.global_min_ if self.global_min_ is not None else 0.0
        elif stat == "max":
            return self.global_max_ if self.global_max_ is not None else 1.0
        return 0.0

    def fit_transform(self, df, y, fold_indices):
        # Compute globals
        self.global_mean_ = float(y.mean())
        self.global_std_ = float(y.std()) if len(y) > 1 else 0.0
        self.global_count_ = float(len(y))
        self.global_min_ = float(y.min())
        self.global_max_ = float(y.max())

        # Initialize result arrays
        results = {stat: pd.Series(np.nan, index=df.index, dtype=float) for stat in self.stats}

        # OOF computation
        for tr_idx, va_idx in fold_indices:
            tr_col = df[self.col].iloc[tr_idx]
            tr_y = y.iloc[tr_idx]

            fold_stats = self._compute_fold_stats(tr_col, tr_y)

            for stat in self.stats:
                if stat in fold_stats:
                    results[stat].iloc[va_idx] = (
                        df[self.col].iloc[va_idx].map(fold_stats[stat])
                    )

        # Fill NaN with neutral defaults
        for stat in self.stats:
            fill_val = self._get_fill_value(stat)
            results[stat] = results[stat].fillna(fill_val)

        # Compute full-data mappings for transform
        self.mappings_ = self._compute_fold_stats(df[self.col], y)

        # Build output DataFrame
        out = {}
        for stat in self.stats:
            out[self._col_name(stat)] = results[stat].values

        return pd.DataFrame(out, index=df.index)

    def transform(self, df):
        out = {}
        for stat in self.stats:
            fill_val = self._get_fill_value(stat)
            if self.mappings_ is not None and stat in self.mappings_:
                vals = df[self.col].map(self.mappings_[stat]).fillna(fill_val)
            else:
                vals = pd.Series(fill_val, index=df.index, dtype=float)
            out[self._col_name(stat)] = vals.values

        return pd.DataFrame(out, index=df.index)


# ============================================================================
# ARITHMETIC INTERACTION EXTENDED (ALL NUMERIC COLUMNS)
# ============================================================================
class ArithmeticInteractionExtended(FeatureGenerator):
    """Sum, difference, product, or ratio between two columns including ordinal integers.

    Identical logic to ``ArithmeticInteraction`` but named distinctly so that
    the pipeline can apply it to the full set of numeric columns — including
    ordinal integer columns that may be excluded from the standard arithmetic
    interaction sweep.

    Parameters
    ----------
    col_a : str
        First column (numeric or ordinal integer).
    col_b : str
        Second column (numeric or ordinal integer).
    op : str
        Operation: ``'add'``, ``'sub'``, ``'mul'``, or ``'div'``.
    """

    def __init__(self, col_a: str, col_b: str, op: str = "add"):
        super().__init__(f"arithext__{col_a}_{op}_{col_b}")
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
        return a + b

    def fit_transform(self, df, y, fold_indices):
        vals = self._compute(df)
        return pd.DataFrame({self.name: vals.values}, index=df.index)

    def transform(self, df):
        vals = self._compute(df)
        return pd.DataFrame({self.name: vals.values}, index=df.index)


# ============================================================================
# PAIR PRODUCT FEATURE (NaN-PROPAGATING)
# ============================================================================
class PairProductFeature(FeatureGenerator):
    """Raw arithmetic product of two columns with NaN propagation.

    Unlike ``PolynomialFeature(poly_type='cross')`` which fills NaN with 0
    before multiplying, this generator propagates NaN so that downstream
    imputation can handle missing values appropriately. It also accepts any
    two column names regardless of whether they are purely continuous or
    ordinal-encoded integers.

    Parameters
    ----------
    col_a : str
        First column (numeric or ordinal integer).
    col_b : str
        Second column (numeric or ordinal integer).
    """

    def __init__(self, col_a: str, col_b: str):
        super().__init__(f"prod__{col_a}_x_{col_b}")
        self.col_a = col_a
        self.col_b = col_b

    def _compute(self, df: pd.DataFrame) -> pd.Series:
        a = df[self.col_a].astype(float)
        b = df[self.col_b].astype(float)
        return a * b

    def fit_transform(self, df, y, fold_indices):
        vals = self._compute(df)
        return pd.DataFrame({self.name: vals.values}, index=df.index)

    def transform(self, df):
        vals = self._compute(df)
        return pd.DataFrame({self.name: vals.values}, index=df.index)
