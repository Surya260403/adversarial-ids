
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def main():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")

    features = [
        c for c in train.columns
        if c not in ["Label", "BinaryLabel"]
    ]

    # Group rows by their complete feature values.
    train_groups = train.groupby(features, dropna=False)["BinaryLabel"].agg(
        train_rows="size",
        train_labels=lambda s: frozenset(s.unique()),
    )

    test_groups = test.groupby(features, dropna=False)["BinaryLabel"].agg(
        test_rows="size",
        test_labels=lambda s: frozenset(s.unique()),
    )

    shared = train_groups.join(test_groups, how="inner")

    conflicts = shared[
        shared.apply(
            lambda r: r["train_labels"].isdisjoint(r["test_labels"]),
            axis=1,
        )
    ]

    print("Distinct shared feature patterns:", len(shared))
    print("Shared patterns with conflicting labels:", len(conflicts))
    print("Shared patterns without conflicting labels:", len(shared) - len(conflicts))
    print("Training rows in shared patterns:", int(shared["train_rows"].sum()))
    print("Testing rows in shared patterns:", int(shared["test_rows"].sum()))

    print("\nFirst 10 conflicts:")
    for _, row in conflicts.head(10).iterrows():
        print(
            f"Train labels={sorted(row['train_labels'])}, "
            f"Test labels={sorted(row['test_labels'])}, "
            f"Train rows={row['train_rows']}, "
            f"Test rows={row['test_rows']}"
        )


if __name__ == "__main__":
    main()
