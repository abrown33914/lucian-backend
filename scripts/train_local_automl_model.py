import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_IN = os.path.join(BASE_DIR, "traffic_automl_training.csv")
MODEL_OUT = os.path.join(BASE_DIR, "delayratio_forecast_rf.joblib")

def main():
    df = pd.read_csv(CSV_IN)

    feat_cols = [
        "delayRatio",
        "currentSpeed",
        "freeFlowSpeed",
        "hour",
        "dow",
        "lat",
        "lon",
    ]

    X = df[feat_cols]
    y = df["delayRatio_future"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    base_model = RandomForestRegressor(random_state=42)

    # Hyperparameter search space (this is your "AutoML-style" part)
    param_dist = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=10,
        cv=3,
        scoring="neg_root_mean_squared_error",
        random_state=42,
        n_jobs=-1,
    )

    print("Running randomized search (AutoML-style)...")
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    print("Best params:", search.best_params_)

    preds = best_model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = mse ** 0.5
    print("Test RMSE:", rmse)

    joblib.dump(best_model, MODEL_OUT)
    print("Saved model to:", MODEL_OUT)

if __name__ == "__main__":
    main()
