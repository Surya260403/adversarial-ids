
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"


def main():
    print("=== 1. LOAD TRAIN / TEST ===")

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")

    features = joblib.load(MODELS / "feature_columns.joblib")
    imputer = joblib.load(MODELS / "baseline_imputer.joblib")
    model = joblib.load(MODELS / "baseline_random_forest.joblib")

    print("Train shape:", train.shape)
    print("Test shape:", test.shape)
    print("Feature count:", len(features))

    assert "BinaryLabel" in train.columns
    assert "BinaryLabel" in test.columns
    assert len(features) == 78
    assert all(c in train.columns and c in test.columns for c in features)

    print("\n=== 2. CLASS BALANCE ===")
    print("Train labels:")
    print(train["BinaryLabel"].value_counts().sort_index())
    print("Test labels:")
    print(test["BinaryLabel"].value_counts().sort_index())

    print("\n=== 3. TRAIN / TEST DUPLICATE CHECK ===")
    # Check exact feature-row overlap without labels.
    train_hash = pd.util.hash_pandas_object(
        train[features], index=False
    )
    test_hash = pd.util.hash_pandas_object(
        test[features], index=False
    )

    overlap = len(set(train_hash).intersection(set(test_hash)))
    print("Matching feature rows across splits:", overlap)

    # Hash collisions are theoretically possible; treat this as a
    # screening check, not a proof that every match is identical.
    print("Note: matches are hash-based; inspect duplicates if nonzero.")

    print("\n=== 4. FEATURE QUALITY ===")
    X_train_df = train[features].apply(pd.to_numeric, errors="coerce")
    X_test_df = test[features].apply(pd.to_numeric, errors="coerce")

    print("Train missing values:", int(X_train_df.isna().sum().sum()))
    print("Test missing values:", int(X_test_df.isna().sum().sum()))
    print("Train infinite values:", int(np.isinf(
        X_train_df.to_numpy(dtype=float)
    ).sum()))
    print("Test infinite values:", int(np.isinf(
        X_test_df.to_numpy(dtype=float)
    ).sum()))

    print("\n=== 5. RE-EVALUATE SAVED BASELINE ===")
    X_test = imputer.transform(X_test_df)
    y_test = test["BinaryLabel"].astype(int)
    predictions = model.predict(X_test)

    print("Accuracy:", round(accuracy_score(y_test, predictions), 6))
    print("Confusion matrix [BENIGN, ATTACK]:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["BENIGN", "ATTACK"],
        zero_division=0,
    ))

    RESULTS.mkdir(parents=True, exist_ok=True)

    report = {
        "train_rows": len(train),
        "test_rows": len(test),
        "feature_count": len(features),
        "matching_feature_hashes": overlap,
        "train_missing_values": int(X_train_df.isna().sum().sum()),
        "test_missing_values": int(X_test_df.isna().sum().sum()),
        "baseline_accuracy": accuracy_score(y_test, predictions),
    }

    output = RESULTS / "validation_report.csv"
    pd.DataFrame([report]).to_csv(output, index=False)

    print("Validation summary saved to:", output)
    print("\nValidation script completed.")


if __name__ == "__main__":
    main()
