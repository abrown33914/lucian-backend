import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_IN = os.path.join(BASE_DIR, "pavement_automl_training.csv")
MODEL_OUT = os.path.join(BASE_DIR, "pavement_model.joblib")


def main():
    df = pd.read_csv(CSV_IN)

    feat_cols = [
        "delayRatio",
        "jamFactor",       # this is your synthetic jam factor now
        "currentSpeed",
        "freeFlowSpeed",
        "hour",
        "dow",
        "lat",
        "lon",
    ]
    target_col = "pavementStressIndex_future"

    X = df[feat_cols].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    base_model = RandomForestRegressor(random_state=42, n_jobs=-1)

    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_grid,
        n_iter=15,
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1,
    )

    print("Running randomized search (AutoML-style) for pavement model...")
    search.fit(X_train, y_train)

    best = search.best_estimator_
    print("Best params:", search.best_params_)

    preds = best.predict(X_test)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    print(f"Pavement model RMSE: {rmse:.4f}")

    joblib.dump(best, MODEL_OUT)
    print("Saved pavement model to:", MODEL_OUT)


if __name__ == "__main__":
    main()
