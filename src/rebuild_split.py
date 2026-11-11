
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def main():
    df = pd.read_csv(DATA / "balanced_dataset.csv")

    features = [
        c for c in df.columns
        if c not in ["Label", "BinaryLabel"]
    ]

    # Group identical feature vectors together, even if their labels differ.
    groups = pd.util.hash_pandas_object(
        df[features],
        index=False,
    )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_idx, test_idx = next(
        splitter.split(df, df["BinaryLabel"], groups=groups)
    )

    train = df.iloc[train_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)

    print("Train shape:", train.shape)
    print("Test shape:", test.shape)
    print("\nTrain labels:")
    print(train["BinaryLabel"].value_counts().sort_index())
    print("\nTest labels:")
    print(test["BinaryLabel"].value_counts().sort_index())

    # Confirm that no feature-vector hash crosses the split.
    train_groups = set(groups.iloc[train_idx])
    test_groups = set(groups.iloc[test_idx])

    print(
        "\nFeature patterns crossing splits:",
        len(train_groups & test_groups),
    )

    train.to_csv(DATA / "train_grouped.csv", index=False)
    test.to_csv(DATA / "test_grouped.csv", index=False)

    print("\nSaved data/train_grouped.csv")
    print("Saved data/test_grouped.csv")
    print("Original train.csv and test.csv were not changed.")


if __name__ == "__main__":
    main()
