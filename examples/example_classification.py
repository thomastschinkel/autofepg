"""
Example: Binary Classification with AutoFE-PG

Demonstrates how to use AutoFE-PG on a synthetic binary classification dataset.
Replace the synthetic data with your actual Kaggle competition data.
"""

import numpy as np
import pandas as pd
from autofepg import select_features


def main():
    # -----------------------------------------------------------------------
    # 1. Generate synthetic data (replace with pd.read_csv("train.csv"), etc.)
    # -----------------------------------------------------------------------
    rng = np.random.RandomState(42)
    n_train = 5000
    n_test = 1000

    def make_data(n):
        return pd.DataFrame(
            {
                "cat_1": rng.choice(["A", "B", "C", "D"], n),
                "cat_2": rng.choice(["X", "Y"], n),
                "num_1": rng.randn(n) * 100,
                "num_2": rng.rand(n) * 50,
                "num_3": rng.randint(100, 99999, n).astype(float),
                "num_4": rng.exponential(10, n),
            }
        )

    X_train = make_data(n_train)
    X_test = make_data(n_test)
    y_train = pd.Series(
        (X_train["num_1"] + X_train["num_2"] * 2 + rng.randn(n_train) * 30 > 50).astype(int),
        name="target",
    )

    # -----------------------------------------------------------------------
    # 2. Run AutoFE-PG
    # -----------------------------------------------------------------------
    result = select_features(
        X_train,
        y_train,
        X_test,
        task="classification",
        n_folds=5,
        time_budget=300,  # 5 minutes
        max_pair_cols=6,
        max_digit_positions=3,
        improvement_threshold=0.0005,
        xgb_params={
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
        },
    )

    # -----------------------------------------------------------------------
    # 3. Inspect results
    # -----------------------------------------------------------------------
    print(f"\nBaseline AUC: {result['base_score']:.6f} ± {result['base_score_std']:.6f}")
    print(f"Best AUC:     {result['best_score']:.6f} ± {result['best_score_std']:.6f}")
    print(f"Features selected: {len(result['selected_features'])}")
    for feat in result["selected_features"]:
        print(f"  - {feat}")

    print(f"\nX_train shape: {result['X_train'].shape}")
    print(f"X_test  shape: {result['X_test'].shape}")

    # -----------------------------------------------------------------------
    # 4. Use the augmented data for final training
    # -----------------------------------------------------------------------
    # X_train_final = result["X_train"]
    # X_test_final = result["X_test"]
    # ... train your final model here ...


if __name__ == "__main__":
    main()
