"""
Feature generators for AutoFE-PG.

Each generator produces one or more columns and implements:
    - ``fit_transform(df, y, fold_indices)`` → pd.DataFrame (train features)
    - ``transform(df)`` → pd.DataFrame (test features)

All generators that use the target ``y`` do so in an out-of-fold manner
to prevent target leakage.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

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
