"""
Example: Regression with AutoFE-PG

Demonstrates how to use AutoFE-PG on a synthetic regression dataset.
Replace the synthetic data with your actual Kaggle competition data.
"""

import numpy as np
import pandas as pd
from autofepg import AutoFE


def main():
    # -----------------------------------------------------------------------
    # 1. Generate synthetic data
    # -----------------------------------------------------------------------
    rng = np.random.RandomState(123)
    n = 3000

    X_train = pd.DataFrame(
        {
            "region": rng.choice(["north", "south", "east", "west"], n),
            "type": rng.choice(["A", "B", "C"], n),
            "area": rng.rand(n) * 200 + 50,
            "age": rng.randint(1, 50, n).astype(float),
            "rooms": rng.randint(1, 8, n),
            "price_per_sqft": rng.rand(n) * 100 + 50,
        }
    )

    y_train = pd.Series(
        X_train["area"] * X_train["price_per_sqft"]
        + X_train["age"] * -500
        + rng.randn(n) * 5000,
        name="price",
    )

    X_test = X_train.head(500).copy()

    # -----------------------------------------------------------------------
    # 2. Run AutoFE-PG with class API
    # -----------------------------------------------------------------------
    autofe = AutoFE(
        task="regression",
        n_folds=5,
        time_budget=180,
        improvement_threshold=1.0,  # MSE needs larger threshold
        backward_selection=True,
        sample=2000,
        xgb_params={
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.1,
        },
    )

    X_train_new, X_test_new = autofe.fit_select(
        X_train,
        y_train,
        X_test,
        cat_cols=["region", "type"],
        num_cols=["area", "age", "rooms", "price_per_sqft"],
    )

    # -----------------------------------------------------------------------
    # 3. Inspect results
    # -----------------------------------------------------------------------
    print(f"\nBaseline MSE: {autofe.base_score_:.2f} ± {autofe.base_score_std_:.2f}")
    print(f"Best MSE:     {autofe.best_score_:.2f} ± {autofe.best_score_std_:.2f}")
    print(f"Features: {autofe.get_selected_feature_names()}")

    history = autofe.get_history()
    print(f"\nHistory shape: {history.shape}")
    print(history.head(10))


if __name__ == "__main__":
    main()
