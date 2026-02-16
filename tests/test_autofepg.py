"""
Tests for the AutoFE-PG library.

Covers utility functions, individual feature generators, the builder,
the CV engine, and end-to-end integration.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold, StratifiedKFold

from autofepg.utils import infer_task, is_improvement, default_metric
from autofepg.generators import (
    ArithmeticInteraction,
    CountEncoding,
    CountEncodingOnDigit,
    CountEncodingOnPair,
    DigitBasePairTE,
    DigitFeature,
    DigitInteraction,
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
from autofepg.builder import FeatureCandidateBuilder
from autofepg.engine import XGBCVEngine
from autofepg.core import AutoFE, select_features


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def classification_data():
    """Small binary classification dataset."""
    rng = np.random.RandomState(42)
    n = 200
    df = pd.DataFrame(
        {
            "cat_a": rng.choice(["x", "y", "z"], n),
            "cat_b": rng.choice(["p", "q"], n),
            "num_a": rng.randn(n) * 100,
            "num_b": rng.rand(n) * 50 + 10,
            "num_c": rng.randint(0, 1000, n).astype(float),
        }
    )
    y = pd.Series(rng.choice([0, 1], n), name="target")
    return df, y


@pytest.fixture
def regression_data():
    """Small regression dataset."""
    rng = np.random.RandomState(42)
    n = 200
    df = pd.DataFrame(
        {
            "cat_a": rng.choice(["a", "b", "c", "d"], n),
            "num_a": rng.randn(n) * 10,
            "num_b": rng.rand(n) * 100,
        }
    )
    y = pd.Series(df["num_a"] * 2 + rng.randn(n) * 0.5, name="target")
    return df, y


@pytest.fixture
def fold_indices(classification_data):
    """5-fold stratified indices."""
    df, y = classification_data
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return list(skf.split(df, y))


@pytest.fixture
def regression_fold_indices(regression_data):
    """5-fold indices for regression."""
    df, y = regression_data
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    return list(kf.split(df, y))


# ============================================================================
# UTILITY TESTS
# ============================================================================


class TestUtils:
    def test_infer_classification_from_object(self):
        y = pd.Series(["a", "b", "a", "c"])
        assert infer_task(y) == "classification"

    def test_infer_classification_from_int(self):
        y = pd.Series([0, 1, 0, 1, 1, 0])
        assert infer_task(y) == "classification"

    def test_infer_regression_from_float(self):
        y = pd.Series([1.1, 2.3, 5.7, 0.4, 3.14])
        assert infer_task(y) == "regression"

    def test_is_improvement_maximize(self):
        assert is_improvement(0.85, 0.84, "maximize", 0.001)
        assert not is_improvement(0.84, 0.84, "maximize", 0.001)
        assert not is_improvement(0.8405, 0.84, "maximize", 0.001)

    def test_is_improvement_minimize(self):
        assert is_improvement(0.10, 0.15, "minimize", 0.001)
        assert not is_improvement(0.15, 0.15, "minimize", 0.001)

    def test_default_metric_classification(self):
        fn, direction = default_metric("classification")
        assert direction == "maximize"
        assert callable(fn)

    def test_default_metric_regression(self):
        fn, direction = default_metric("regression")
        assert direction == "minimize"
        assert callable(fn)


# ============================================================================
# GENERATOR TESTS
# ============================================================================


class TestPairInteraction:
    def test_fit_transform_and_transform(self, classification_data, fold_indices):
        df, y = classification_data
        gen = PairInteraction("cat_a", "cat_b")
        result = gen.fit_transform(df, y, fold_indices)
        assert result.shape == (len(df), 1)
        assert gen.name in result.columns

        test_df = df.head(10).copy()
        test_result = gen.transform(test_df)
        assert test_result.shape == (10, 1)


class TestTargetEncoding:
    def test_no_leakage(self, classification_data, fold_indices):
        df, y = classification_data
        gen = TargetEncoding("cat_a")
        result = gen.fit_transform(df, y, fold_indices)
        assert result.shape == (len(df), 1)
        assert not result.isna().any().any()

    def test_transform(self, classification_data, fold_indices):
        df, y = classification_data
        gen = TargetEncoding("cat_a")
        gen.fit_transform(df, y, fold_indices)
        test_result = gen.transform(df.head(5))
        assert test_result.shape == (5, 1)


class TestCountEncoding:
    def test_basic(self, classification_data, fold_indices):
        df, y = classification_data
        gen = CountEncoding("cat_a")
        result = gen.fit_transform(df, y, fold_indices)
        assert result.shape == (len(df), 1)
        assert (result.values >= 0).all()


class TestDigitFeature:
    def test_extraction(self):
        df = pd.DataFrame({"x": [123, 456, 789, 0, 12345]})
        y = pd.Series([0, 1, 0, 1, 0])
        folds = [(np.array([0, 1, 2]), np.array([3, 4]))]

        gen = DigitFeature("x", 0)
        result = gen.fit_transform(df, y, folds)
        assert list(result.values.flatten()) == [3, 6, 9, 0, 5]

        gen2 = DigitFeature("x", 1)
        result2 = gen2.fit_transform(df, y, folds)
        assert list(result2.values.flatten()) == [2, 5, 8, 0, 4]


class TestDigitInteraction:
    def test_basic(self):
        df = pd.DataFrame({"a": [123, 456], "b": [789, 321]})
        y = pd.Series([0, 1])
        folds = [(np.array([0]), np.array([1]))]

        gen = DigitInteraction([("a", 0), ("b", 0)])
        result = gen.fit_transform(df, y, folds)
        assert result.shape == (2, 1)


class TestRoundFeature:
    def test_rounding(self):
        df = pd.DataFrame({"x": [3.14159, 2.71828, 1.41421]})
        y = pd.Series([0, 1, 0])
        folds = [(np.array([0, 1]), np.array([2]))]

        gen = RoundFeature("x", 2)
        result = gen.fit_transform(df, y, folds)
        np.testing.assert_array_almost_equal(
            result.values.flatten(), [3.14, 2.72, 1.41]
        )


class TestQuantileBinFeature:
    def test_basic(self, classification_data, fold_indices):
        df, y = classification_data
        gen = QuantileBinFeature("num_a", n_bins=5)
        result = gen.fit_transform(df, y, fold_indices)
        assert result.shape == (len(df), 1)


class TestNumToCat:
    def test_basic(self, classification_data, fold_indices):
        df, y = classification_data
        gen = NumToCat("num_a", n_bins=5)
        result = gen.fit_transform(df, y, fold_indices)
        assert result.shape == (len(df), 1)


class TestArithmeticInteraction:
    def test_all_ops(self, classification_data, fold_indices):
        df, y = classification_data
        for op in ["add", "sub", "mul", "div"]:
            gen = ArithmeticInteraction("num_a", "num_b", op)
            result = gen.fit_transform(df, y, fold_indices)
            assert result.shape == (len(df), 1)


class TestFrequencyEncoding:
    def test_basic(self, classification_data, fold_indices):
        df, y = classification_data
        gen = FrequencyEncoding("cat_a")
        result = gen.fit_transform(df, y, fold_indices)
        assert result.shape == (len(df), 1)
        assert (result.values >= 0).all()
        assert (result.values <= 1).all()


class TestMissingIndicator:
    def test_with_missing(self):
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0, np.nan, 5.0]})
        y = pd.Series([0, 1, 0, 1, 0])
        folds = [(np.array([0, 1, 2]), np.array([3, 4]))]

        gen = MissingIndicator("x")
        result = gen.fit_transform(df, y, folds)
        assert list(result.values.flatten()) == [0, 1, 0, 1, 0]


class TestGroupStatFeature:
    def test_basic(self, classification_data, fold_indices):
        df, y = classification_data
        gen = GroupStatFeature("num_a", "cat_a", "mean")
        result = gen.fit_transform(df, y, fold_indices)
        assert result.shape == (len(df), 1)


class TestGroupDeviationFeature:
    def test_basic(self, classification_data, fold_indices):
        df, y = classification_data
        gen = GroupDeviationFeature("num_a", "cat_a", "diff")
        result = gen.fit_transform(df, y, fold_indices)
        assert result.shape == (len(df), 1)


class TestUnaryTransform:
    def test_all_types(self, classification_data, fold_indices):
        df, y = classification_data
        for ttype in ["log1p", "sqrt", "square", "reciprocal"]:
            gen = UnaryTransform("num_a", ttype)
            result = gen.fit_transform(df, y, fold_indices)
            assert result.shape == (len(df), 1)


class TestPolynomialFeature:
    def test_square(self, classification_data, fold_indices):
        df, y = classification_data
        gen = PolynomialFeature("num_a", poly_type="square")
        result = gen.fit_transform(df, y, fold_indices)
        assert result.shape == (len(df), 1)

    def test_cross(self, classification_data, fold_indices):
        df, y = classification_data
        gen = PolynomialFeature("num_a", "num_b", poly_type="cross")
        result = gen.fit_transform(df, y, fold_indices)
        assert result.shape == (len(df), 1)


class TestTargetEncodingAuxTarget:
    def test_basic(self, classification_data, fold_indices):
        df, y = classification_data
        gen = TargetEncodingAuxTarget("cat_a", "num_a")
        result = gen.fit_transform(df, y, fold_indices)
        assert result.shape == (len(df), 1)


class TestCompositeGenerators:
    def test_te_on_pair(self, classification_data, fold_indices):
        df, y = classification_data
        gen = TargetEncodingOnPair("cat_a", "cat_b")
        result = gen.fit_transform(df.copy(), y, fold_indices)
        assert result.shape == (len(df), 1)

        test_result = gen.transform(df.head(5).copy())
        assert test_result.shape == (5, 1)

    def test_ce_on_pair(self, classification_data, fold_indices):
        df, y = classification_data
        gen = CountEncodingOnPair("cat_a", "cat_b")
        result = gen.fit_transform(df.copy(), y, fold_indices)
        assert result.shape == (len(df), 1)

    def test_te_on_digit(self, classification_data, fold_indices):
        df, y = classification_data
        gen = TargetEncodingOnDigit("num_c", 0)
        result = gen.fit_transform(df.copy(), y, fold_indices)
        assert result.shape == (len(df), 1)

    def test_ce_on_digit(self, classification_data, fold_indices):
        df, y = classification_data
        gen = CountEncodingOnDigit("num_c", 0)
        result = gen.fit_transform(df.copy(), y, fold_indices)
        assert result.shape == (len(df), 1)

    def test_digit_base_pair_te(self, classification_data, fold_indices):
        df, y = classification_data
        gen = DigitBasePairTE("num_c", 0, "cat_a")
        result = gen.fit_transform(df.copy(), y, fold_indices)
        assert result.shape == (len(df), 1)


# ============================================================================
# BUILDER TESTS
# ============================================================================


class TestFeatureCandidateBuilder:
    def test_auto_detect_columns(self, classification_data):
        df, _ = classification_data
        cat_cols, num_cols = FeatureCandidateBuilder.auto_detect_columns(df)
        assert "cat_a" in cat_cols
        assert "cat_b" in cat_cols
        assert "num_a" in num_cols or "num_b" in num_cols

    def test_build_produces_candidates(self, classification_data):
        df, y = classification_data
        builder = FeatureCandidateBuilder(max_pair_cols=5, max_digit_positions=2)
        candidates = builder.build(df, y)
        assert len(candidates) > 0
        assert all(hasattr(c, "fit_transform") for c in candidates)
        assert all(hasattr(c, "transform") for c in candidates)


# ============================================================================
# ENGINE TESTS
# ============================================================================


class TestXGBCVEngine:
    def test_classification_evaluate(self, classification_data):
        df, y = classification_data
        engine = XGBCVEngine(
            task="classification",
            n_folds=3,
            xgb_params={"n_estimators": 10, "max_depth": 3},
        )
        folds = engine.get_fold_indices(df, y)
        mean_score, std_score = engine.evaluate(df, y, folds)
        assert isinstance(mean_score, float)
        assert isinstance(std_score, float)
        assert 0 <= mean_score <= 1
        assert std_score >= 0

    def test_regression_evaluate(self, regression_data):
        df, y = regression_data
        engine = XGBCVEngine(
            task="regression",
            n_folds=3,
            xgb_params={"n_estimators": 10, "max_depth": 3},
        )
        folds = engine.get_fold_indices(df, y)
        mean_score, std_score = engine.evaluate(df, y, folds)
        assert isinstance(mean_score, float)
        assert mean_score >= 0


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestAutoFEIntegration:
    def test_classification_end_to_end(self, classification_data):
        df, y = classification_data
        test_df = df.head(20).copy()

        autofe = AutoFE(
            task="classification",
            n_folds=3,
            time_budget=60,
            verbose=False,
            max_pair_cols=3,
            max_digit_positions=1,
            xgb_params={"n_estimators": 10, "max_depth": 3},
        )

        X_train_new, X_test_new = autofe.fit_select(df, y, test_df)

        assert X_train_new.shape[0] == len(df)
        assert X_test_new.shape[0] == 20
        assert X_train_new.shape[1] >= df.shape[1]
        assert autofe.base_score_ is not None
        assert autofe.best_score_ is not None

    def test_regression_end_to_end(self, regression_data):
        df, y = regression_data

        autofe = AutoFE(
            task="regression",
            n_folds=3,
            time_budget=60,
            verbose=False,
            max_pair_cols=3,
            max_digit_positions=1,
            xgb_params={"n_estimators": 10, "max_depth": 3},
        )

        X_train_new = autofe.fit_select(df, y)
        assert X_train_new.shape[0] == len(df)

    def test_select_features_convenience(self, classification_data):
        df, y = classification_data

        result = select_features(
            df,
            y,
            task="classification",
            n_folds=3,
            time_budget=30,
            verbose=False,
            max_pair_cols=3,
            max_digit_positions=1,
            xgb_params={"n_estimators": 10, "max_depth": 3},
        )

        assert "X_train" in result
        assert "autofe" in result
        assert "history" in result
        assert "selected_features" in result
        assert "base_score" in result
        assert "best_score" in result

    def test_sampling(self, classification_data):
        df, y = classification_data

        autofe = AutoFE(
            task="classification",
            n_folds=3,
            time_budget=30,
            verbose=False,
            max_pair_cols=3,
            max_digit_positions=1,
            sample=50,
            xgb_params={"n_estimators": 10, "max_depth": 3},
        )

        X_train_new = autofe.fit_select(df, y)
        # Final refit is on full data
        assert X_train_new.shape[0] == len(df)

    def test_transform_method(self, classification_data):
        df, y = classification_data

        autofe = AutoFE(
            task="classification",
            n_folds=3,
            time_budget=30,
            verbose=False,
            max_pair_cols=3,
            max_digit_positions=1,
            xgb_params={"n_estimators": 10, "max_depth": 3},
        )

        autofe.fit_select(df, y)
        new_data = df.head(10).copy()
        transformed = autofe.transform(new_data)
        assert transformed.shape[0] == 10

    def test_get_history(self, classification_data):
        df, y = classification_data

        autofe = AutoFE(
            task="classification",
            n_folds=3,
            time_budget=30,
            verbose=False,
            max_pair_cols=3,
            max_digit_positions=1,
            xgb_params={"n_estimators": 10, "max_depth": 3},
        )

        autofe.fit_select(df, y)
        history = autofe.get_history()
        assert isinstance(history, pd.DataFrame)
        if len(history) > 0:
            assert "name" in history.columns
            assert "kept" in history.columns
