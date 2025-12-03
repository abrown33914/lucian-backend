import os
import pandas as pd
import numpy as np

# Resolve CSV paths relative to this script file so running from repo root works
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_IN = os.path.join(BASE_DIR, "traffic_history_fgcu.csv")
CSV_OUT = os.path.join(BASE_DIR, "traffic_automl_training.csv")

def main():
    try:
        df = pd.read_csv(CSV_IN)
    except FileNotFoundError:
        raise FileNotFoundError(f"Input CSV not found at {CSV_IN}. Place the CSV there or update CSV_IN.")

    # Parse timestamp & sort so we can make "future" labels
    df["snapshotTimestamp"] = pd.to_datetime(df["snapshotTimestamp"])
    df = df.sort_values(["twinId", "snapshotTimestamp"])

    # Basic numeric cleaning
    for col in ["currentSpeed", "freeFlowSpeed",
                "currentTravelTime", "freeFlowTravelTime"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Delay ratio now
    df["delayRatio"] = df["currentTravelTime"] / df["freeFlowTravelTime"]
    df["delayRatio"] = df["delayRatio"].replace([np.inf, -np.inf], np.nan)

    # Target: delay ratio at next 5-minute step (same road)
    df["delayRatio_future"] = df.groupby("twinId")["delayRatio"].shift(-1)

    # Time features for AutoML
    df["hour"] = df["snapshotTimestamp"].dt.hour
    df["dow"] = df["snapshotTimestamp"].dt.dayofweek  # 0=Mon

    # Drop rows with missing target or key features
    feat_cols = [
        "delayRatio",
        "currentSpeed",
        "freeFlowSpeed",
        "hour",
        "dow",
        "lat",
        "lon",
    ]
    keep_cols = ["snapshotTimestamp", "twinId"] + feat_cols + ["delayRatio_future"]

    df = df[keep_cols].dropna(subset=feat_cols + ["delayRatio_future"])

    print("Rows after cleaning:", len(df))
    df.to_csv(CSV_OUT, index=False)
    print("Wrote:", CSV_OUT)

if __name__ == "__main__":
    main()
