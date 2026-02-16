"""
Feature candidate builder for AutoFE-PG.

Generates an ordered list of feature engineering candidates covering all strategies,
prioritized by expected impact.
"""

import pandas as pd
from itertools import combinations
from typing import List, Optional, Tuple

from autofepg.generators import (
    ArithmeticInteraction,
    CountEncoding,
    CountEncodingOnDigit,
    CountEncodingOnPair,
    DigitBasePairTE,
    DigitFeature,
    DigitInteraction,
    FeatureGenerator,
    FrequencyEncoding,
    GroupDeviationFeature,
    GroupStatFeature,
    MissingIndicator,
    NumToCat,
    PairInteraction,
    PolynomialFeature,
    QuantileBinFeature,
    RoundFeature,
    TargetEncoding,
    TargetEncodingAuxTarget,
    TargetEncodingOnDigit,
    TargetEncodingOnPair,
    UnaryTransform,
)


class FeatureCandidateBuilder:
    """Build a prioritized list of feature generator candidates.

    Given a DataFrame, auto-detects column types and produces candidates
    from a hardcoded sequence ordered by expected impact:

    1.  Target Encoding (single columns)
    2.  Count Encoding (single columns)
    3.  Target Encoding on pairs
    4.  Count Encoding on pairs
    5.  Frequency Encoding
    6.  Missing Indicators
    7.  TE with auxiliary targets
    8.  Unary transforms (log, sqrt, square, reciprocal)
    9.  Arithmetic interactions
    10. Polynomial features
    11. Pairwise label-encoded interactions
    12. TE/CE on digits
    13. Digit × Cat TE
    14. Quantile binning
    15. Digit features
    16. Digit interactions (within & across)
    17. Rounding features
    18. Num-to-Cat conversion
    19. GroupStat & GroupDeviation (aggregation stats)

    Parameters
    ----------
    cat_cols : list of str, optional
        Categorical columns. Auto-detected if None.
    num_cols : list of str, optional
        Numerical columns. Auto-detected if None.
    aux_target_cols : list of str, optional
        Columns to use as auxiliary targets for TE.
    max_pair_cols : int
        Max columns to consider for pairwise features.
    max_digit_positions : int
        Max digit positions to extract.
    max_digit_interaction_order : int
        Max order for digit interactions.
    rounding_decimals : list of int, optional
        Decimal places for rounding features.
    quantile_bins : list of int, optional
        Number of bins for quantile binning.
    """

    def __init__(
        self,
        cat_cols: Optional[List[str]] = None,
        num_cols: Optional[List[str]] = None,
        aux_target_cols: Optional[List[str]] = None,
        max_pair_cols: int = 30,
        max_digit_positions: int = 4,
        max_digit_interaction_order: int = 3,
        rounding_decimals: Optional[List[int]] = None,
        quantile_bins: Optional[List[int]] = None,
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
    def auto_detect_columns(
        df: pd.DataFrame, target_col: Optional[str] = None
    ) -> Tuple[List[str], List[str]]:
        """Auto-detect categorical and numerical columns.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        target_col : str, optional
            Target column name to exclude.

        Returns
        -------
        tuple of (list, list)
            ``(categorical_columns, numerical_columns)``
        """
        cat_cols, num_cols = [], []
        for c in df.columns:
            if target_col and c == target_col:
                continue
            if c.lower() == "id":
                continue
            if df[c].dtype == "object" or df[c].dtype.name == "category":
                cat_cols.append(c)
            elif (
                df[c].nunique() <= 30
                and df[c].dtype in ["int64", "int32", "int16", "int8"]
            ):
                cat_cols.append(c)
            else:
                num_cols.append(c)
        return cat_cols, num_cols

    def build(self, df: pd.DataFrame, y: pd.Series) -> List[FeatureGenerator]:
        """Build the complete list of feature generator candidates.

        Parameters
        ----------
        df : pd.DataFrame
            Training features.
        y : pd.Series
            Training target.

        Returns
        -------
        list of FeatureGenerator
            Ordered list of candidate generators.
        """
        if self.cat_cols is None or self.num_cols is None:
            auto_cat, auto_num = self.auto_detect_columns(df)
            if self.cat_cols is None:
                self.cat_cols = auto_cat
            if self.num_cols is None:
                self.num_cols = auto_num

        all_cols = self.cat_cols + self.num_cols
        candidates: List[FeatureGenerator] = []

        print(
            f"[AutoFE-PG] Detected {len(self.cat_cols)} cat cols, "
            f"{len(self.num_cols)} num cols"
        )
        print(
            f"[AutoFE-PG] Cat: {self.cat_cols[:10]}"
            f"{'...' if len(self.cat_cols) > 10 else ''}"
        )
        print(
            f"[AutoFE-PG] Num: {self.num_cols[:10]}"
            f"{'...' if len(self.num_cols) > 10 else ''}"
        )

        pair_cols = all_cols[: self.max_pair_cols]

        # 1) TE of base features
        for c in all_cols:
            candidates.append(TargetEncoding(c))

        # 2) CE of base features
        for c in all_cols:
            candidates.append(CountEncoding(c))

        # 3) TE of pair interactions
        for a, b in combinations(pair_cols, 2):
            candidates.append(TargetEncodingOnPair(a, b))

        # 4) CE of pair interactions
        for a, b in combinations(pair_cols, 2):
            candidates.append(CountEncodingOnPair(a, b))

        # 5) Frequency encoding
        for c in all_cols:
            candidates.append(FrequencyEncoding(c))

        # 6) Missing indicator features
        cols_with_missing = [c for c in all_cols if df[c].isna().any()]
        for c in cols_with_missing:
            candidates.append(MissingIndicator(c))

        # 7) TE with auxiliary targets
        for aux_col in self.aux_target_cols:
            if aux_col in df.columns:
                for c in all_cols:
                    candidates.append(TargetEncodingAuxTarget(c, aux_col))

        # 8) Unary transforms
        for c in self.num_cols:
            for ttype in ["log1p", "sqrt", "square", "reciprocal"]:
                candidates.append(UnaryTransform(c, ttype))

        # 9) Arithmetic interactions
        num_arith = self.num_cols[: min(len(self.num_cols), 15)]
        for a, b in combinations(num_arith, 2):
            for op in ["add", "sub", "mul", "div"]:
                candidates.append(ArithmeticInteraction(a, b, op))

        # 10) Polynomial features
        num_poly = self.num_cols[: min(len(self.num_cols), 15)]
        for c in num_poly:
            candidates.append(PolynomialFeature(c, poly_type="square"))
        for a, b in combinations(num_poly, 2):
            candidates.append(PolynomialFeature(a, b, poly_type="cross"))

        # 11) Pairwise label-encoded interactions
        for a, b in combinations(pair_cols, 2):
            candidates.append(PairInteraction(a, b))

        # 12) TE/CE on digit features
        for c in self.num_cols[:10]:
            for d in range(min(self.max_digit_positions, 3)):
                candidates.append(TargetEncodingOnDigit(c, d))
                candidates.append(CountEncodingOnDigit(c, d))

        # 13) Digit x Cat TE
        for c_num in self.num_cols[:10]:
            for c_cat in self.cat_cols[:10]:
                candidates.append(DigitBasePairTE(c_num, 0, c_cat))

        # 14) Quantile binning
        for c in self.num_cols:
            for nb in self.quantile_bins:
                candidates.append(QuantileBinFeature(c, n_bins=nb))

        # 15) Digit features
        for c in self.num_cols:
            for d in range(self.max_digit_positions):
                candidates.append(DigitFeature(c, d))

        # 16) Digit interactions WITHIN same feature
        for c in self.num_cols:
            digit_positions = list(range(min(self.max_digit_positions, 4)))
            for order in range(
                2, min(self.max_digit_interaction_order + 1, len(digit_positions) + 1)
            ):
                for combo in combinations(digit_positions, order):
                    candidates.append(DigitInteraction([(c, d) for d in combo]))

        # 17) Digit interactions ACROSS features
        num_limited = self.num_cols[: min(len(self.num_cols), 10)]
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

        # 18) Rounding features
        for c in self.num_cols:
            for dec in self.rounding_decimals:
                candidates.append(RoundFeature(c, dec))

        # 19) Num-to-Cat conversion
        for c in self.num_cols:
            candidates.append(NumToCat(c, n_bins=5))
            candidates.append(NumToCat(c, n_bins=10))

        # 20) Group statistics & deviations
        for c_num in self.num_cols[:15]:
            for c_cat in self.cat_cols[:15]:
                for stat in ["mean", "std", "min", "max", "median"]:
                    candidates.append(GroupStatFeature(c_num, c_cat, stat))
                for mode in ["diff", "ratio"]:
                    candidates.append(GroupDeviationFeature(c_num, c_cat, mode))

        print(f"[AutoFE-PG] Total candidates generated: {len(candidates)}")
        return candidates
