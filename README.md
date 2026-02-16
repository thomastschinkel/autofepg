# 🧪 AutoFE-PG

**Automatic Feature Engineering & Selection for Kaggle Playground Competitions**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/yourusername/autofepg/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/autofepg/actions/workflows/ci.yml)

AutoFE-PG is a production-ready library that **automatically generates, evaluates, and selects** engineered features to boost your tabular ML models — with zero target leakage.

---

## ✨ Key Features

| Feature                         | Description                                                                                            |
| ------------------------------- | ------------------------------------------------------------------------------------------------------ |
| **Auto column detection**       | Automatically identifies categorical vs. numerical columns                                             |
| **20+ feature strategies**      | Target encoding, count encoding, digit extraction, arithmetic interactions, group statistics, and more |
| **Zero target leakage**         | All target-dependent features use strict out-of-fold encoding                                          |
| **Greedy forward selection**    | Adds features one-by-one, keeping only those that improve CV score                                     |
| **Optional backward pruning**   | Removes redundant features after forward selection                                                     |
| **GPU acceleration**            | Automatically uses XGBoost GPU if available                                                            |
| **Time budget**                 | Set a wall-clock limit; the search stops gracefully                                                    |
| **Sampling support**            | Evaluate on a subsample for faster iteration                                                           |
| **Custom XGBoost params**       | Pass your own hyperparameters                                                                          |
| **Score variance tracking**     | Reports mean ± std across folds                                                                        |
| **Classification & regression** | Supports both tasks with auto-detection                                                                |

---

## 🚀 Quick Start

### Installation

```bash
pip install autofepg .
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

### Using the Class API

```python
from autofepg import AutoFE

autofe = AutoFE(
    task="classification",
    n_folds=5,
    time_budget=1800,
    improvement_threshold=0.0001,
    backward_selection=True,
    sample=10000,
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
```

---

## 📖 How It Works

### 1. Feature Generation

AutoFE-PG generates candidates from a hardcoded priority sequence ordered by expected impact:

| Priority | Strategy                            | Leakage-free?     |
| -------- | ----------------------------------- | ----------------- |
| 1        | Target Encoding (single columns)    | ✅ OOF             |
| 2        | Count Encoding (single columns)     | ✅ No target       |
| 3        | Target Encoding on pairs            | ✅ OOF             |
| 4        | Count Encoding on pairs             | ✅ No target       |
| 5        | Frequency Encoding                  | ✅ No target       |
| 6        | Missing Indicators                  | ✅ No target       |
| 7        | TE with auxiliary targets           | ✅ OOF             |
| 8        | Unary transforms (log, sqrt, etc.)  | ✅ No target       |
| 9        | Arithmetic interactions             | ✅ No target       |
| 10       | Polynomial features                 | ✅ No target       |
| 11       | Pairwise label-encoded interactions | ✅ No target       |
| 12       | TE/CE on digit features             | ✅ OOF / No target |
| 13       | Digit × Category TE                 | ✅ OOF             |
| 14       | Quantile binning                    | ✅ No target       |
| 15       | Raw digit extraction                | ✅ No target       |
| 16       | Digit interactions                  | ✅ No target       |
| 17       | Rounding features                   | ✅ No target       |
| 18       | Num-to-Cat conversion               | ✅ No target       |
| 19       | Group statistics & deviations       | ✅ No target       |

### 2. Greedy Forward Selection

Each candidate is evaluated by adding it to the current feature set and running XGBoost K-fold CV.
A feature is kept only if it improves the score beyond the configured threshold.

### 3. Optional Backward Pruning

After forward selection, features are tested for removal.
If removing a feature improves (or maintains) the score, it is permanently dropped.

---

## ⚙️ Configuration

| Parameter             | Type     | Default | Description                               |
| --------------------- | -------- | ------- | ----------------------------------------- |
| task                  | str      | "auto"  | "classification", "regression", or "auto" |
| n_folds               | int      | 5       | Number of CV folds                        |
| time_budget           | float    | None    | Max seconds (wall clock)                  |
| improvement_threshold | float    | 1e-7    | Min score delta to keep a feature         |
| sample                | int      | None    | Subsample rows for faster CV              |
| backward_selection    | bool     | False   | Run backward pruning after forward        |
| max_pair_cols         | int      | 20      | Max columns for pairwise features         |
| max_digit_positions   | int      | 4       | Max digit positions to extract            |
| xgb_params            | dict     | None    | Custom XGBoost hyperparameters            |
| metric_fn             | callable | None    | Custom metric (y_true, y_pred) -> float   |
| metric_direction      | str      | None    | "maximize" or "minimize"                  |
| random_state          | int      | 42      | Random seed                               |
| verbose               | bool     | True    | Print progress                            |

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

```
autofepg/
├── autofepg/
│   ├── __init__.py          # Public API
│   ├── utils.py             # GPU detection, task inference, metrics
│   ├── generators.py        # All feature generator classes
│   ├── builder.py           # FeatureCandidateBuilder
│   ├── engine.py            # XGBoost CV engine
│   └── core.py              # AutoFE class + select_features()
├── tests/
│   ├── __init__.py
│   └── test_autofepg.py     # Unit and integration tests
├── examples/
│   ├── example_classification.py
│   └── example_regression.py
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
