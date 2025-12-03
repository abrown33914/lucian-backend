import os
import pandas as pd
import numpy as np

# Resolve CSV paths relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# This is the CSV exported from /traffic/history/export
CSV_IN = os.path.join(BASE_DIR, "traffic_history_fgcu.csv")
CSV_OUT = os.path.join(BASE_DIR, "pavement_automl_training.csv")


def main():
    try:
        df = pd.read_csv(CSV_IN)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Input CSV not found at {CSV_IN}. "
            "Call the /traffic/history/export endpoint and save the file there."
        )

    # Parse timestamp and sort so we can do 'future' labels
    df["snapshotTimestamp"] = pd.to_datetime(df["snapshotTimestamp"])
    df = df.sort_values(["twinId", "snapshotTimestamp"])

    # Numeric cleaning
    numeric_cols = [
        "currentSpeed",
        "freeFlowSpeed",
        "jamFactor",
        "currentTravelTime",
        "freeFlowTravelTime",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Build synthetic jam factor using speed drop
    df["pseudoJam"] = 10 * (1 - (df["currentSpeed"] / df["freeFlowSpeed"]))
    df["pseudoJam"] = df["pseudoJam"].clip(lower=0, upper=10)

    # Replace jamFactor with pseudo if all zeros or NaN
    df["jamFactor"] = df["pseudoJam"].fillna(0.0)

    # Delay ratio now
    df["delayRatio"] = df["currentTravelTime"] / df["freeFlowTravelTime"]
    df["delayRatio"] = df["delayRatio"].replace([np.inf, -np.inf], np.nan)

    # Simple pavement stress index NOW:
    #   0.5 * delayRatio  +  0.5 * (jamFactor / 10)
    # Jam factor is 0–10, so jamFactor/10 is 0–1
    df["pavementStressIndex"] = 0.6 * df["delayRatio"] + 0.4 * (df["jamFactor"] / 10.0)

    # Target: pavement stress at the NEXT 5-minute snapshot for the same segment
    df["pavementStressIndex_future"] = (
        df.groupby("twinId")["pavementStressIndex"].shift(-1)
    )

    # Time features
    df["hour"] = df["snapshotTimestamp"].dt.hour
    df["dow"] = df["snapshotTimestamp"].dt.dayofweek  # 0=Mon

    # Features we want to train on
    feat_cols = [
        "delayRatio",
        "jamFactor",
        "currentSpeed",
        "freeFlowSpeed",
        "hour",
        "dow",
        "lat",
        "lon",
    ]

    keep_cols = (
        ["snapshotTimestamp", "twinId"]
        + feat_cols
        + ["pavementStressIndex", "pavementStressIndex_future"]
    )

    df = df[keep_cols].dropna(
        subset=feat_cols + ["pavementStressIndex", "pavementStressIndex_future"]
    )

    print("Rows after cleaning:", len(df))
    df.to_csv(CSV_OUT, index=False)
    print("Wrote:", CSV_OUT)


if __name__ == "__main__":
    main()
