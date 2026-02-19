# 🧪 AutoFE-PG

**Automatic Feature Engineering & Selection for Kaggle Playground Competitions**

![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![CI](https://img.shields.io/badge/CI-passing-brightgreen)

AutoFE-PG is a production-ready library that automatically generates, evaluates, and selects engineered features to boost your tabular ML models — with zero target leakage.

Designed specifically for Kaggle Playground competitions where synthetic data is common, it includes specialized strategies for **domain alignment**, **Bayesian priors from external data**, **dual-representation features**, and **cross-dataset density analysis**.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| Auto column detection | Automatically identifies categorical vs. numerical columns |
| 25+ feature strategies | Target encoding, domain alignment, Bayesian priors, dual representation, cross-dataset frequency, count encoding, digit extraction, arithmetic interactions, group statistics, and more |
| Zero target leakage | All target-dependent features use strict out-of-fold encoding |
| Greedy forward selection | Adds features one-by-one, keeping only those that improve CV score |
| Optional backward pruning | Removes redundant features after forward selection |
| Original data integration | Snap synthetic values to real clinical grids and inject historical priors |
| GPU acceleration | Automatically uses XGBoost GPU if available |
| Time budget | Set a wall-clock limit; the search stops gracefully |
| Sampling support | Evaluate on a subsample for faster iteration |
| Custom XGBoost params | Pass your own hyperparameters |
| Score variance tracking | Reports mean ± std across folds |
| Classification & regression | Supports both tasks with auto-detection |
| Detailed reports | Auto-generated `.txt` report with full selection history |

---

## 🚀 Quick Start

### Installation

```bash
pip install autofepg
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

### Minimal Example

```python
import pandas as pd
from autofepg import select_features

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

X_train = train.drop(columns=["id", "target"])
y_train = train["target"]
X_test = test.drop(columns=["id"])

result = select_features(
    X_train, y_train, X_test,
    task="classification",
    time_budget=3600,
)

X_train_new = result["X_train"]
X_test_new = result["X_test"]

print(f"Baseline AUC: {result['base_score']:.6f}")
print(f"Best AUC:     {result['best_score']:.6f}")
print(f"Features added: {len(result['selected_features'])}")
```

### With Original Data (Domain Alignment + Bayesian Priors)

When working with Kaggle Playground competitions where synthetic data is generated from a real dataset, you can pass the original data to unlock powerful de-noising and prior-injection strategies:

```python
import pandas as pd
from autofepg import select_features

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
original = pd.read_csv("original.csv")

X_train = train.drop(columns=["id", "target"])
y_train = train["target"]
X_test = test.drop(columns=["id"])

X_original = original.drop(columns=["target"])
y_original = original["target"]

result = select_features(
    X_train, y_train, X_test,
    task="classification",
    time_budget=3600,
    original_df=X_original,
    original_target=y_original,
)

X_train_new = result["X_train"]
X_test_new = result["X_test"]

print(f"Baseline AUC: {result['base_score']:.6f}")
print(f"Best AUC:     {result['best_score']:.6f}")
print(f"Features added: {len(result['selected_features'])}")
```

### Using the Class API

```python
from autofepg import AutoFE
import pandas as pd

original = pd.read_csv("original.csv")

autofe = AutoFE(
    task="classification",
    n_folds=5,
    time_budget=1800,
    improvement_threshold=0.0001,
    backward_selection=True,
    sample=10000,
    original_df=original.drop(columns=["target"]),
    original_target=original["target"],
    xgb_params={
        "n_estimators": 1000,
        "max_depth": 8,
        "learning_rate": 0.05,
    },
)

X_train_new, X_test_new = autofe.fit_select(
    X_train, y_train, X_test,
    aux_target_cols=["employment_status", "debt_to_income_ratio"],
)

# Inspect results
print(autofe.get_selected_feature_names())
history_df = autofe.get_history()
details_df = autofe.get_selection_details()
```

---

## 📖 How It Works

### 1. Feature Generation

AutoFE-PG generates candidates from a hardcoded priority sequence ordered by expected impact:

| Priority | Strategy | Description | Leakage-free? |
|---|---|---|---|
| 1 | **Domain Alignment** | Snap synthetic values to nearest real-data grid point; expose residual | ✅ No target |
| 2 | **Bayesian Priors** | Inject P(target \| value) from original dataset as external knowledge | ✅ No train target |
| 3 | Target Encoding (single) | OOF mean-target per category | ✅ OOF |
| 4 | Count Encoding (single) | Value counts per category | ✅ No target |
| 5 | **Dual Representation** | Continuous + label-encoded copy of each numerical column | ✅ No target |
| 6 | Target Encoding on pairs | OOF TE on column pair interactions | ✅ OOF |
| 7 | Count Encoding on pairs | Value counts on column pair interactions | ✅ No target |
| 8 | Frequency Encoding | Normalized value counts | ✅ No target |
| 9 | **Cross-Dataset Frequency & Rarity** | How common/rare a value is across train+test+original | ✅ No target |
| 10 | Missing Indicators | Binary NaN flags | ✅ No target |
| 11 | TE with auxiliary targets | OOF TE using a different column as target | ✅ OOF |
| 12 | Unary transforms | log1p, sqrt, square, reciprocal | ✅ No target |
| 13 | Arithmetic interactions | add, sub, mul, div between numerical pairs | ✅ No target |
| 14 | Polynomial features | Square and cross-product terms | ✅ No target |
| 15 | Pairwise label interactions | Label-encoded column pairs | ✅ No target |
| 16 | TE/CE on digits | Target/count encoding on extracted digits | ✅ OOF / No target |
| 17 | Digit × Category TE | Digit-category interaction with OOF TE | ✅ OOF |
| 18 | Quantile binning | Equal-frequency bins | ✅ No target |
| 19 | Raw digit extraction | i-th digit of numerical values | ✅ No target |
| 20 | Digit interactions | Within-feature and cross-feature digit combos | ✅ No target |
| 21 | Rounding features | Round to various decimal places / magnitudes | ✅ No target |
| 22 | Num-to-Cat conversion | Equal-width binning | ✅ No target |
| 23 | Group statistics & deviations | Mean, std, min, max, median by group; diff/ratio to group | ✅ No target |

### 2. Greedy Forward Selection

Each candidate is evaluated by adding it to the current feature set and running XGBoost K-fold CV.
A feature is kept only if it improves the score beyond the configured threshold.

### 3. Optional Backward Pruning

After forward selection, features are tested for removal.
If removing a feature improves (or maintains) the score, it is permanently dropped.

---

## 🧬 Synthetic Data Strategies

AutoFE-PG includes four strategies specifically designed for Kaggle Playground competitions where the training data is synthetically generated from a real-world dataset.

### Domain Alignment (De-noising)

The synthetic generation process often introduces "fuzzy" values that wouldn't exist in a real clinical setting. **Domain Alignment** forces every continuous value in the synthetic set to its nearest neighbor in the original dataset, effectively "snapping" the data back to its true clinical grid. The residual (distance to the snap point) is also exposed as a feature, since it encodes how much the synthetic process perturbed the value.

```python
from autofepg.generators import DomainAlignmentFeature
import numpy as np

# Reference values from original dataset
ref_vals = original["blood_pressure"].dropna().unique()
gen = DomainAlignmentFeature("blood_pressure", reference_values=ref_vals)
```

### Bayesian-Style Priors (External Mapping)

Instead of letting the model learn strictly from the training data, **Bayesian Priors** import external knowledge from the original dataset. By calculating `P(target | value)` in the original file and injecting those probabilities as features, the model starts with a "hint" about which values are clinically dangerous. This uses no information from the training target — zero leakage.

```python
from autofepg.generators import BayesianPriorFeature

# Pre-computed from original data
prior_map = original.groupby("cholesterol")["heart_disease"].mean().to_dict()
gen = BayesianPriorFeature("cholesterol", prior_map=prior_map)
```

### Dimensionality Expansion (Dual Representation)

The model uses a "dual-representation" strategy for numerical features:

- **Continuous copy**: Treated as a number to capture linear or threshold trends
- **Categorical copy**: Treated as a discrete label-encoded value to allow the tree to create very specific, non-linear splits on exact values

```python
from autofepg.generators import DualRepresentationFeature

gen = DualRepresentationFeature("age")
# Produces: dual__age_cont (float) + dual__age_cat (int label)
```

### Frequency and Density Analysis

Cross-dataset frequency analysis calculates the rarity of values across the entire data ecosystem (train, test, and original). This helps the model identify if a specific data point is an outlier or part of a common cluster — a strong signal in synthetic datasets where certain "modes" are over-represented.

```python
from autofepg.generators import CrossDatasetFrequencyFeature, ValueRarityFeature
import pandas as pd

# Combine counts across all datasets
combined = pd.concat([train["age"], test["age"], original["age"]])
eco_counts = combined.value_counts()
eco_total = len(combined)

freq_gen = CrossDatasetFrequencyFeature("age", eco_counts, eco_total)
rare_gen = ValueRarityFeature("age", eco_counts, eco_total)
```

> **Note:** When you pass `original_df`, `original_target`, and `X_test` to `AutoFE` or `select_features`, all four strategies are automatically generated and evaluated. No manual setup required.

---

## ⚙️ Configuration

| Parameter | Type | Default | Description |
|---|---|---|---|
| `task` | str | `"auto"` | `"classification"`, `"regression"`, or `"auto"` |
| `n_folds` | int | `5` | Number of CV folds |
| `time_budget` | float | `None` | Max seconds (wall clock) |
| `improvement_threshold` | float | `1e-7` | Min score delta to keep a feature |
| `sample` | int | `None` | Subsample rows for faster CV |
| `backward_selection` | bool | `False` | Run backward pruning after forward |
| `max_pair_cols` | int | `20` | Max columns for pairwise features |
| `max_digit_positions` | int | `4` | Max digit positions to extract |
| `xgb_params` | dict | `None` | Custom XGBoost hyperparameters |
| `metric_fn` | callable | `None` | Custom metric `(y_true, y_pred) -> float` |
| `metric_direction` | str | `None` | `"maximize"` or `"minimize"` |
| `random_state` | int | `42` | Random seed |
| `verbose` | bool | `True` | Print progress |
| `original_df` | DataFrame | `None` | Original (real) dataset features for domain alignment & priors |
| `original_target` | Series | `None` | Original dataset target for Bayesian prior computation |
| `report_path` | str | `"autofepg_report.txt"` | Path for detailed selection report |

---

## 📊 Output

The `select_features()` function returns a dictionary:

```python
{
    "X_train": pd.DataFrame,          # Augmented training data
    "X_test": pd.DataFrame,           # Augmented test data (if provided)
    "autofe": AutoFE,                 # Fitted AutoFE object
    "history": pd.DataFrame,          # Full selection history
    "selected_features": List[str],   # Names of kept features
    "selection_details": pd.DataFrame, # Per-feature improvement details
    "base_score": float,              # Baseline CV mean
    "base_score_std": float,          # Baseline CV std
    "best_score": float,              # Final CV mean
    "best_score_std": float,          # Final CV std
}
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📁 Project Structure

```text
autofepg/
├── autofepg/
│   ├── __init__.py          # Public API & exports
│   ├── utils.py             # GPU detection, task inference, metrics
│   ├── generators.py        # All feature generator classes (25+)
│   ├── builder.py           # FeatureCandidateBuilder
│   ├── engine.py            # XGBoost CV engine
│   └── core.py              # AutoFE class + select_features()
├── tests/
│   ├── __init__.py
│   └── test_autofepg.py     # Unit and integration tests
├── examples/
│   ├── example_classification.py
│   ├── example_regression.py
│   └── example_with_original.py
├── .github/
│   └── workflows/
│       └── ci.yml
├── .gitignore
├── LICENSE
├── README.md
├── CHANGELOG.md
├── CONTRIBUTING.md
├── Makefile
├── pyproject.toml
├── setup.py
└── requirements.txt
```

---

## 📋 Generator Reference

### Original Strategies

| Generator | Class | Target used? |
|---|---|---|
| Target Encoding | `TargetEncoding` | ✅ OOF |
| Count Encoding | `CountEncoding` | ❌ |
| Frequency Encoding | `FrequencyEncoding` | ❌ |
| Pair Interaction | `PairInteraction` | ❌ |
| TE on Pairs | `TargetEncodingOnPair` | ✅ OOF |
| CE on Pairs | `CountEncodingOnPair` | ❌ |
| Digit Extraction | `DigitFeature` | ❌ |
| Digit Interaction | `DigitInteraction` | ❌ |
| TE on Digits | `TargetEncodingOnDigit` | ✅ OOF |
| CE on Digits | `CountEncodingOnDigit` | ❌ |
| Digit × Cat TE | `DigitBasePairTE` | ✅ OOF |
| Rounding | `RoundFeature` | ❌ |
| Quantile Binning | `QuantileBinFeature` | ❌ |
| Num-to-Cat | `NumToCat` | ❌ |
| TE with Aux Target | `TargetEncodingAuxTarget` | ✅ OOF (aux) |
| Arithmetic Interaction | `ArithmeticInteraction` | ❌ |
| Missing Indicator | `MissingIndicator` | ❌ |
| Group Statistics | `GroupStatFeature` | ❌ |
| Group Deviation | `GroupDeviationFeature` | ❌ |
| Unary Transform | `UnaryTransform` | ❌ |
| Polynomial Feature | `PolynomialFeature` | ❌ |

### Synthetic Data Strategies (NEW in v0.2.0)

| Generator | Class | Requires | Target used? |
|---|---|---|---|
| Domain Alignment | `DomainAlignmentFeature` | `original_df` | ❌ |
| Bayesian Prior | `BayesianPriorFeature` | `original_df` + `original_target` | ❌ (external only) |
| Dual Representation | `DualRepresentationFeature` | — | ❌ |
| Cross-Dataset Frequency | `CrossDatasetFrequencyFeature` | `original_df` or `X_test` | ❌ |
| Value Rarity | `ValueRarityFeature` | `original_df` or `X_test` | ❌ |

---

## 📝 Changelog

### v0.2.0

- **Domain Alignment**: Snap synthetic values to nearest real-data grid point with residual feature
- **Bayesian Priors**: Inject external P(target|value) from original dataset
- **Dual Representation**: Continuous + categorical copy of numerical features
- **Cross-Dataset Frequency**: Value frequency across train+test+original ecosystem
- **Value Rarity**: Log-inverse-frequency score for outlier detection
- Added `original_df` and `original_target` parameters to `AutoFE` and `select_features`
- Report now includes original data status
- Version bump to 0.2.0

### v0.1.3

- Initial public release
- 20+ feature generation strategies
- Greedy forward selection with optional backward pruning
- GPU acceleration support
- Detailed text report generation

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
