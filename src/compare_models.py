
"""
Compare four IDS algorithms:
1. Existing grouped Random Forest
2. XGBoost
3. Linear SVM
4. Neural Network (MLP)

Existing project artifacts are loaded but never overwritten.
"""

from pathlib import Path
import time
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

MODELS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

TRAIN_FILE = DATA / "train_grouped.csv"
TEST_FILE = DATA / "test_grouped.csv"

RF_FILE = MODELS / "grouped_baseline_random_forest.joblib"
RF_IMPUTER_FILE = MODELS / "grouped_baseline_imputer.joblib"
FEATURE_FILE = MODELS / "grouped_feature_columns.joblib"

TARGET = "BinaryLabel"
RANDOM_STATE = 42

# --------------------------------------------------
# 2. Load grouped training and testing data
# --------------------------------------------------

print("\nLoading grouped datasets...")

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

if TARGET not in train_df.columns or TARGET not in test_df.columns:
    raise ValueError("BinaryLabel column is missing.")

y_train = train_df[TARGET].astype(int)
y_test = test_df[TARGET].astype(int)

if not set(y_train.unique()).issubset({0, 1}):
    raise ValueError("Training target must contain only 0 and 1.")

if not set(y_test.unique()).issubset({0, 1}):
    raise ValueError("Testing target must contain only 0 and 1.")

# Load the exact feature list used by the grouped RF.
feature_columns = joblib.load(FEATURE_FILE)

if isinstance(feature_columns, np.ndarray):
    feature_columns = feature_columns.tolist()

feature_columns = list(feature_columns)

if "Label" in feature_columns or TARGET in feature_columns:
    raise ValueError("Target leakage detected in feature list.")

missing_train = set(feature_columns) - set(train_df.columns)
missing_test = set(feature_columns) - set(test_df.columns)

if missing_train or missing_test:
    raise ValueError(
        f"Missing features. Train: {missing_train}; Test: {missing_test}"
    )

X_train = train_df[feature_columns].apply(
    pd.to_numeric, errors="coerce"
)
X_test = test_df[feature_columns].apply(
    pd.to_numeric, errors="coerce"
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Features:", len(feature_columns))

# --------------------------------------------------
# 3. Load and prepare existing Random Forest
# --------------------------------------------------

print("\nLoading existing Random Forest...")

rf_model = joblib.load(RF_FILE)
rf_imputer = joblib.load(RF_IMPUTER_FILE)

X_train_rf = rf_imputer.transform(X_train)
X_test_rf = rf_imputer.transform(X_test)

# --------------------------------------------------
# 4. Prepare preprocessing for SVM and MLP
# --------------------------------------------------

print("\nPreparing scaled data for SVM and MLP...")

svm_imputer = SimpleImputer(strategy="median")
X_train_svm_imputed = svm_imputer.fit_transform(X_train)
X_test_svm_imputed = svm_imputer.transform(X_test)

svm_scaler = StandardScaler()
X_train_scaled = svm_scaler.fit_transform(X_train_svm_imputed)
X_test_scaled = svm_scaler.transform(X_test_svm_imputed)

# --------------------------------------------------
# 5. Common evaluation function
# --------------------------------------------------

comparison_results = []
confusion_results = []
errors = []


def evaluate_model(name, model, X_eval, training_time):
    """Evaluate one model on the common test dataset."""

    print(f"\nEvaluating {name}...")

    start = time.perf_counter()
    predictions = model.predict(X_eval)
    prediction_seconds = time.perf_counter() - start

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    metrics = {
        "Algorithm": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1_Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "False_Positive_Rate": fpr,
        "Training_Time_Seconds": training_time,
        "Prediction_Time_ms_per_1000_samples": (
            prediction_seconds / len(y_test) * 1_000_000
        )
    }

    comparison_results.append(metrics)

    confusion_results.append({
        "Algorithm": name,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    })

    print(
        f"{name}: Accuracy={metrics['Accuracy']:.4f}, "
        f"Precision={metrics['Precision']:.4f}, "
        f"Recall={metrics['Recall']:.4f}, "
        f"F1={metrics['F1_Score']:.4f}"
    )


# --------------------------------------------------
# 6. Evaluate existing Random Forest
# --------------------------------------------------

try:
    # Training time is unavailable because this model
    # was trained in an earlier experiment.
    evaluate_model(
        "Random Forest",
        rf_model,
        X_test_rf,
        np.nan
    )
except Exception as exc:
    errors.append(("Random Forest", str(exc)))
    print("Random Forest error:", exc)


# --------------------------------------------------
# 7. Train and evaluate XGBoost
# --------------------------------------------------

try:
    print("\nTraining XGBoost...")

    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    start = time.perf_counter()
    xgb_model.fit(X_train_rf, y_train)
    xgb_training_time = time.perf_counter() - start

    joblib.dump(xgb_model, MODELS / "xgboost_model.joblib")

    evaluate_model(
        "XGBoost",
        xgb_model,
        X_test_rf,
        xgb_training_time
    )

except Exception as exc:
    errors.append(("XGBoost", str(exc)))
    print("XGBoost error:", exc)


# --------------------------------------------------
# 8. Train and evaluate Linear SVM
# --------------------------------------------------

try:
    print("\nTraining Linear SVM...")

    svm_model = LinearSVC(
        C=1.0,
        max_iter=5000,
        dual=False,
        random_state=RANDOM_STATE
    )

    start = time.perf_counter()
    svm_model.fit(X_train_scaled, y_train)
    svm_training_time = time.perf_counter() - start

    joblib.dump(svm_model, MODELS / "svm_model.joblib")
    joblib.dump(svm_imputer, MODELS / "svm_imputer.joblib")
    joblib.dump(svm_scaler, MODELS / "svm_scaler.joblib")

    evaluate_model(
        "Linear SVM",
        svm_model,
        X_test_scaled,
        svm_training_time
    )

except Exception as exc:
    errors.append(("Linear SVM", str(exc)))
    print("Linear SVM error:", exc)


# --------------------------------------------------
# 9. Train and evaluate Neural Network (MLP)
# --------------------------------------------------

try:
    print("\nTraining Neural Network (MLP)...")

    mlp_imputer = SimpleImputer(strategy="median")
    X_train_mlp_imputed = mlp_imputer.fit_transform(X_train)
    X_test_mlp_imputed = mlp_imputer.transform(X_test)

    mlp_scaler = StandardScaler()
    X_train_mlp = mlp_scaler.fit_transform(X_train_mlp_imputed)
    X_test_mlp = mlp_scaler.transform(X_test_mlp_imputed)

    mlp_model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        batch_size=1024,
        learning_rate_init=0.001,
        max_iter=30,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        random_state=RANDOM_STATE,
        verbose=True
    )

    start = time.perf_counter()
    mlp_model.fit(X_train_mlp, y_train)
    mlp_training_time = time.perf_counter() - start

    joblib.dump(mlp_model, MODELS / "mlp_model.joblib")
    joblib.dump(mlp_imputer, MODELS / "mlp_imputer.joblib")
    joblib.dump(mlp_scaler, MODELS / "mlp_scaler.joblib")

    evaluate_model(
        "Neural Network (MLP)",
        mlp_model,
        X_test_mlp,
        mlp_training_time
    )

except Exception as exc:
    errors.append(("Neural Network (MLP)", str(exc)))
    print("Neural Network error:", exc)


# --------------------------------------------------
# 10. Save comparison results
# --------------------------------------------------

if comparison_results:
    comparison_df = pd.DataFrame(comparison_results)

    comparison_df.to_csv(
        RESULTS / "model_comparison.csv",
        index=False
    )

    confusion_df = pd.DataFrame(confusion_results)

    confusion_df.to_csv(
        RESULTS / "model_confusion_matrices.csv",
        index=False
    )

    print("\n========== MODEL COMPARISON ==========")
    print(comparison_df.to_string(index=False))

    print("\n========== CONFUSION MATRICES ==========")
    print(confusion_df.to_string(index=False))

else:
    print("\nNo models completed successfully.")

# --------------------------------------------------
# 11. Report errors and completion
# --------------------------------------------------

if errors:
    print("\n========== MODEL ERRORS ==========")
    for model_name, error_message in errors:
        print(f"{model_name}: {error_message}")

print("\nComparison process completed.")
print("Results folder:", RESULTS)
print("New model artifacts folder:", MODELS)
print("Existing model artifacts were not intentionally modified.")
