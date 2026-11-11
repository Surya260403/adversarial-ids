from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "balanced_dataset.csv"


def main():
    """Inspect the network-flow features."""

    df = pd.read_csv(DATA_PATH)

    feature_columns = [
        column
        for column in df.columns
        if column not in ["Label", "BinaryLabel"]
    ]

    print("========== FEATURE INSPECTION ==========")
    print("Dataset shape:", df.shape)
    print("Number of features:", len(feature_columns))

    print("\nFeature names:")
    for index, column in enumerate(feature_columns, start=1):
        print(f"{index:02d}. {column}")

    print("\nFeature data types:")
    print(df[feature_columns].dtypes.value_counts())

    print("\nFeature statistics:")
    print(
        df[feature_columns]
        .describe()
        .T[["min", "max", "mean", "std"]]
    )

    print("\nMissing values:")
    missing = df[feature_columns].isna().sum()
    print(missing[missing > 0])


if __name__ == "__main__":
    main()