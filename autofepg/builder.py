"""
Feature candidate builder for AutoFE-PG.

Generates an ordered list of feature engineering candidates covering all strategies,
prioritized by expected impact.
"""

import numpy as np
import pandas as pd
from itertools import combinations
from typing import Dict, List, Optional, Tuple

from autofepg.generators import (
    ArithmeticInteraction,
    BayesianPriorFeature,
    CountEncoding,
    CountEncodingOnDigit,
    CountEncodingOnPair,
    CrossDatasetFrequencyFeature,
    DigitBasePairTE,
    DigitFeature,
    DigitInteraction,
    DomainAlignmentFeature,
    DualRepresentationFeature,
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
    ValueRarityFeature,
)


class FeatureCandidateBuilder:
    """Build a prioritized list of feature generator candidates.

    Given a DataFrame, auto-detects column types and produces candidates
    from a hardcoded sequence ordered by expected impact:

    1.  Domain Alignment (de-noising) — snap to real-data grid
    2.  Bayesian-Style Priors — external target probability hints
    3.  Target Encoding (single columns)
    4.  Count Encoding (single columns)
    5.  Dimensionality Expansion — dual continuous+categorical representation
    6.  Target Encoding on pairs
    7.  Count Encoding on pairs
    8.  Frequency Encoding
    9.  Cross-Dataset Frequency & Value Rarity
    10. Missing Indicators
    11. TE with auxiliary targets
    12. Unary transforms (log, sqrt, square, reciprocal)
    13. Arithmetic interactions
    14. Polynomial features
    15. Pairwise label-encoded interactions
    16. TE/CE on digits
    17. Digit × Cat TE
    18. Quantile binning
    19. Digit features
    20. Digit interactions (within & across)
    21. Rounding features
    22. Num-to-Cat conversion
    23. GroupStat & GroupDeviation (aggregation stats)

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
    original_df : pd.DataFrame, optional
        Original (real-world) dataset for domain alignment and Bayesian priors.
    original_target : pd.Series, optional
        Target column from the original dataset (for Bayesian prior computation).
    test_df : pd.DataFrame, optional
        Test dataset for cross-dataset frequency/density analysis.
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
        original_df: Optional[pd.DataFrame] = None,
        original_target: Optional[pd.Series] = None,
        test_df: Optional[pd.DataFrame] = None,
    ):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.aux_target_cols = aux_target_cols or []
        self.max_pair_cols = max_pair_cols
        self.max_digit_positions = max_digit_positions
        self.max_digit_interaction_order = max_digit_interaction_order
        self.rounding_decimals = rounding_decimals or [-2, -1, 0, 1, 2]
        self.quantile_bins = quantile_bins or [5, 10, 20]
        self.original_df = original_df
        self.original_target = original_target
        self.test_df = test_df

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

    def _build_ecosystem_counts(
        self, col: str, train_df: pd.DataFrame
    ) -> Tuple[pd.Series, int]:
        """Combine value counts across train, test, and original for one column."""
        parts = [train_df[col].value_counts()]
        total = len(train_df)

        if self.test_df is not None and col in self.test_df.columns:
            parts.append(self.test_df[col].value_counts())
            total += len(self.test_df)

        if self.original_df is not None and col in self.original_df.columns:
            parts.append(self.original_df[col].value_counts())
            total += len(self.original_df)

        combined = parts[0]
        for p in parts[1:]:
            combined = combined.add(p, fill_value=0)

        return combined, total

    def _build_prior_map(self, col: str) -> Optional[Dict]:
        """Compute P(target=1 | value) from the original dataset for one column."""
        if self.original_df is None or self.original_target is None:
            return None
        if col not in self.original_df.columns:
            return None

        orig_col = self.original_df[col]
        orig_y = self.original_target

        if len(orig_col) != len(orig_y):
            return None

        try:
            prior = orig_y.groupby(orig_col).mean()
            return prior.to_dict()
        except Exception:
            return None

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

        has_original = self.original_df is not None
        has_original_target = has_original and self.original_target is not None
        has_ecosystem = has_original or self.test_df is not None

        if has_original:
            print(
                f"[AutoFE-PG] Original dataset provided: "
                f"{len(self.original_df)} rows, {len(self.original_df.columns)} cols"
            )
        if has_original_target:
            print("[AutoFE-PG] Original target provided — Bayesian priors enabled")
        if has_ecosystem:
            print("[AutoFE-PG] Cross-dataset frequency/density analysis enabled")

        pair_cols = all_cols[: self.max_pair_cols]

        # ----------------------------------------------------------------
        # 1) Domain Alignment (De-noising) — snap to nearest real value
        # ----------------------------------------------------------------
        if has_original:
            for c in self.num_cols:
                if c in self.original_df.columns:
                    ref_vals = self.original_df[c].dropna().unique()
                    if len(ref_vals) > 0 and len(ref_vals) < 100_000:
                        candidates.append(
                            DomainAlignmentFeature(
                                c, reference_values=ref_vals, include_residual=True
                            )
                        )

        # ----------------------------------------------------------------
        # 2) Bayesian-Style Priors (External Mapping)
        # ----------------------------------------------------------------
        if has_original_target:
            for c in all_cols:
                prior_map = self._build_prior_map(c)
                if prior_map and len(prior_map) > 0:
                    candidates.append(BayesianPriorFeature(c, prior_map))

        # ----------------------------------------------------------------
        # 3) TE of base features
        # ----------------------------------------------------------------
        for c in all_cols:
            candidates.append(TargetEncoding(c))

        # ----------------------------------------------------------------
        # 4) CE of base features
        # ----------------------------------------------------------------
        for c in all_cols:
            candidates.append(CountEncoding(c))

        # ----------------------------------------------------------------
        # 5) Dimensionality Expansion (Dual Representation)
        # ----------------------------------------------------------------
        for c in self.num_cols:
            candidates.append(DualRepresentationFeature(c))

        # ----------------------------------------------------------------
        # 6) TE of pair interactions
        # ----------------------------------------------------------------
        for a, b in combinations(pair_cols, 2):
            candidates.append(TargetEncodingOnPair(a, b))

        # ----------------------------------------------------------------
        # 7) CE of pair interactions
        # ----------------------------------------------------------------
        for a, b in combinations(pair_cols, 2):
            candidates.append(CountEncodingOnPair(a, b))

        # ----------------------------------------------------------------
        # 8) Frequency encoding
        # ----------------------------------------------------------------
        for c in all_cols:
            candidates.append(FrequencyEncoding(c))

        # ----------------------------------------------------------------
        # 9) Cross-Dataset Frequency & Value Rarity
        # ----------------------------------------------------------------
        if has_ecosystem:
            for c in all_cols:
                try:
                    eco_counts, eco_total = self._build_ecosystem_counts(c, df)
                    candidates.append(
                        CrossDatasetFrequencyFeature(c, eco_counts, eco_total)
                    )
                    candidates.append(
                        ValueRarityFeature(c, eco_counts, eco_total)
                    )
                except Exception:
                    pass

        # ----------------------------------------------------------------
        # 10) Missing indicator features
        # ----------------------------------------------------------------
        cols_with_missing = [c for c in all_cols if df[c].isna().any()]
        for c in cols_with_missing:
            candidates.append(MissingIndicator(c))

        # ----------------------------------------------------------------
        # 11) TE with auxiliary targets
        # ----------------------------------------------------------------
        for aux_col in self.aux_target_cols:
            if aux_col in df.columns:
                for c in all_cols:
                    candidates.append(TargetEncodingAuxTarget(c, aux_col))

        # ----------------------------------------------------------------
        # 12) Unary transforms
        # ----------------------------------------------------------------
        for c in self.num_cols:
            for ttype in ["log1p", "sqrt", "square", "reciprocal"]:
                candidates.append(UnaryTransform(c, ttype))

        # ----------------------------------------------------------------
        # 13) Arithmetic interactions
        # ----------------------------------------------------------------
        num_arith = self.num_cols[: min(len(self.num_cols), 15)]
        for a, b in combinations(num_arith, 2):
            for op in ["add", "sub", "mul", "div"]:
                candidates.append(ArithmeticInteraction(a, b, op))

        # ----------------------------------------------------------------
        # 14) Polynomial features
        # ----------------------------------------------------------------
        num_poly = self.num_cols[: min(len(self.num_cols), 15)]
        for c in num_poly:
            candidates.append(PolynomialFeature(c, poly_type="square"))
        for a, b in combinations(num_poly, 2):
            candidates.append(PolynomialFeature(a, b, poly_type="cross"))

        # ----------------------------------------------------------------
        # 15) Pairwise label-encoded interactions
        # ----------------------------------------------------------------
        for a, b in combinations(pair_cols, 2):
            candidates.append(PairInteraction(a, b))

        # ----------------------------------------------------------------
        # 16) TE/CE on digit features
        # ----------------------------------------------------------------
        for c in self.num_cols[:10]:
            for d in range(min(self.max_digit_positions, 3)):
                candidates.append(TargetEncodingOnDigit(c, d))
                candidates.append(CountEncodingOnDigit(c, d))

        # ----------------------------------------------------------------
        # 17) Digit x Cat TE
        # ----------------------------------------------------------------
        for c_num in self.num_cols[:10]:
            for c_cat in self.cat_cols[:10]:
                candidates.append(DigitBasePairTE(c_num, 0, c_cat))

        # ----------------------------------------------------------------
        # 18) Quantile binning
        # ----------------------------------------------------------------
        for c in self.num_cols:
            for nb in self.quantile_bins:
                candidates.append(QuantileBinFeature(c, n_bins=nb))

        # ----------------------------------------------------------------
        # 19) Digit features
        # ----------------------------------------------------------------
        for c in self.num_cols:
            for d in range(self.max_digit_positions):
                candidates.append(DigitFeature(c, d))

        # ----------------------------------------------------------------
        # 20) Digit interactions WITHIN same feature
        # ----------------------------------------------------------------
        for c in self.num_cols:
            digit_positions = list(range(min(self.max_digit_positions, 4)))
            for order in range(
                2, min(self.max_digit_interaction_order + 1, len(digit_positions) + 1)
            ):
                for combo in combinations(digit_positions, order):
                    candidates.append(DigitInteraction([(c, d) for d in combo]))

        # Digit interactions ACROSS features
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

        # ----------------------------------------------------------------
        # 21) Rounding features
        # ----------------------------------------------------------------
        for c in self.num_cols:
            for dec in self.rounding_decimals:
                candidates.append(RoundFeature(c, dec))

        # ----------------------------------------------------------------
        # 22) Num-to-Cat conversion
        # ----------------------------------------------------------------
        for c in self.num_cols:
            candidates.append(NumToCat(c, n_bins=5))
            candidates.append(NumToCat(c, n_bins=10))

        # ----------------------------------------------------------------
        # 23) Group statistics & deviations
        # ----------------------------------------------------------------
        for c_num in self.num_cols[:15]:
            for c_cat in self.cat_cols[:15]:
                for stat in ["mean", "std", "min", "max", "median"]:
                    candidates.append(GroupStatFeature(c_num, c_cat, stat))
                for mode in ["diff", "ratio"]:
                    candidates.append(GroupDeviationFeature(c_num, c_cat, mode))

        print(f"[AutoFE-PG] Total candidates generated: {len(candidates)}")
        return candidates
