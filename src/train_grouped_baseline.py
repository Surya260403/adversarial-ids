
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"


def main():
    train = pd.read_csv(DATA / "train_grouped.csv")
    test = pd.read_csv(DATA / "test_grouped.csv")

    features = [
        c for c in train.columns
        if c not in ["Label", "BinaryLabel"]
    ]

    X_train = train[features].apply(pd.to_numeric, errors="coerce")
    X_test = test[features].apply(pd.to_numeric, errors="coerce")

    y_train = train["BinaryLabel"].astype(int)
    y_test = test["BinaryLabel"].astype(int)

    # Fit preprocessing only on training data.
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    print("Training Random Forest on grouped split...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
    }

    print("\n=== GROUPED-SPLIT BASELINE ===")
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["BENIGN", "ATTACK"],
        zero_division=0,
    ))

    MODELS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODELS / "grouped_baseline_random_forest.joblib")
    joblib.dump(imputer, MODELS / "grouped_baseline_imputer.joblib")
    joblib.dump(features, MODELS / "grouped_feature_columns.joblib")

    pd.DataFrame([metrics]).to_csv(
        RESULTS / "grouped_baseline_metrics.csv",
        index=False,
    )

    print("\nSaved grouped baseline model and metrics.")


if __name__ == "__main__":
    main()
