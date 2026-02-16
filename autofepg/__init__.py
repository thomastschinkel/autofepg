"""
AutoFE - Playground (autofepg)
==============================

Automatic Feature Engineering & Selection for Kaggle Playground Competitions.

Generate, evaluate, and select engineered features with zero target leakage.

Quick Start
-----------
>>> from autofepg import select_features
>>> result = select_features(X_train, y_train, X_test, task="classification")
>>> X_train_new = result["X_train"]
>>> X_test_new = result["X_test"]

Class API
---------
>>> from autofepg import AutoFE
>>> autofe = AutoFE(task="classification", time_budget=3600)
>>> X_train_new, X_test_new = autofe.fit_select(X_train, y_train, X_test)
"""

from autofepg.core import AutoFE, select_features
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
from autofepg.builder import FeatureCandidateBuilder
from autofepg.engine import XGBCVEngine

__version__ = "0.1.1"

__all__ = [
    # Main API
    "AutoFE",
    "select_features",
    # Building blocks
    "FeatureCandidateBuilder",
    "XGBCVEngine",
    # Generators
    "FeatureGenerator",
    "TargetEncoding",
    "CountEncoding",
    "FrequencyEncoding",
    "PairInteraction",
    "TargetEncodingOnPair",
    "CountEncodingOnPair",
    "DigitFeature",
    "DigitInteraction",
    "TargetEncodingOnDigit",
    "CountEncodingOnDigit",
    "DigitBasePairTE",
    "RoundFeature",
    "QuantileBinFeature",
    "NumToCat",
    "TargetEncodingAuxTarget",
    "ArithmeticInteraction",
    "MissingIndicator",
    "GroupStatFeature",
    "GroupDeviationFeature",
    "UnaryTransform",
    "PolynomialFeature",
]
