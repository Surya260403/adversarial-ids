
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "test_grouped.csv"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

MODEL_PATH = MODEL_DIR / "grouped_baseline_random_forest.joblib"
IMPUTER_PATH = MODEL_DIR / "grouped_baseline_imputer.joblib"
FEATURES_PATH = MODEL_DIR / "grouped_feature_columns.joblib"

PERTURBATION_LEVELS = [0.01, 0.03, 0.05, 0.10]
RANDOM_STATE = 42


def main():
    print("Loading grouped baseline artifacts...")

    model = joblib.load(MODEL_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    features = joblib.load(FEATURES_PATH)

    df = pd.read_csv(DATA_PATH)

    X_df = df[features].apply(pd.to_numeric, errors="coerce")
    y = df["BinaryLabel"].astype(int).to_numpy()

    # Use the imputer fitted on training data.
    X = imputer.transform(X_df)

    # Only malicious test rows are used for the evasion-rate analysis.
    attack_mask = y == 1
    X_attack = X[attack_mask]
    y_attack = y[attack_mask]

    print("Test dataset:", df.shape)
    print("Features:", len(features))
    print("Attack samples:", len(X_attack))

    # Bounds are estimated from the training set, not the test set.
    train_df = pd.read_csv(ROOT / "data" / "train_grouped.csv")
    X_train_df = train_df[features].apply(pd.to_numeric, errors="coerce")
    X_train = imputer.transform(X_train_df)

    lower = np.nanpercentile(X_train, 1, axis=0)
    upper = np.nanpercentile(X_train, 99, axis=0)

    # Replace infinite bounds if any feature is constant or degenerate.
    lower = np.nan_to_num(lower, nan=0.0, neginf=0.0, posinf=0.0)
    upper = np.nan_to_num(upper, nan=0.0, neginf=0.0, posinf=0.0)

    # Baseline on clean attack samples.
    clean_predictions = model.predict(X_attack)
    clean_matrix = confusion_matrix(y_attack, clean_predictions, labels=[0, 1])
    clean_evasion = clean_matrix[1, 0] / len(y_attack)

    rows = [{
        "perturbation_strength": 0.0,
        "accuracy_on_attack_samples": accuracy_score(y_attack, clean_predictions),
        "attack_detection_rate": clean_matrix[1, 1] / len(y_attack),
        "evasion_rate": clean_evasion,
        "attack_misclassified_as_benign": int(clean_matrix[1, 0]),
        "f1_score": f1_score(y_attack, clean_predictions, zero_division=0),
    }]

    print("\nClean attack detection rate:", round(1 - clean_evasion, 6))

    # This is a synthetic random stress test, not a validated real-world attack.
    rng = np.random.default_rng(RANDOM_STATE)

    for strength in PERTURBATION_LEVELS:
        print(f"\nTesting {strength:.0%} perturbation...")

        # Perturb all 78 features proportionally to their training scale.
        scale = np.nanpercentile(X_train, 75, axis=0) - np.nanpercentile(
            X_train, 25, axis=0
        )
        scale = np.maximum(np.abs(scale), 1.0)

        directions = rng.choice([-1.0, 1.0], size=X_attack.shape)
        perturbed = X_attack + directions * strength * scale

        perturbed = np.clip(perturbed, lower, upper)

        predictions = model.predict(perturbed)
        matrix = confusion_matrix(y_attack, predictions, labels=[0, 1])

        detection_rate = matrix[1, 1] / len(y_attack)
        evasion_rate = matrix[1, 0] / len(y_attack)

        row = {
            "perturbation_strength": strength,
            "accuracy_on_attack_samples": accuracy_score(y_attack, predictions),
            "attack_detection_rate": detection_rate,
            "evasion_rate": evasion_rate,
            "attack_misclassified_as_benign": int(matrix[1, 0]),
            "f1_score": f1_score(y_attack, predictions, zero_division=0),
        }
        rows.append(row)

        print("Attack detection rate:", round(detection_rate, 6))
        print("Evasion rate:", round(evasion_rate, 6))
        print("ATTACK -> BENIGN:", int(matrix[1, 0]))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "grouped_adversarial_evaluation.csv"

    pd.DataFrame(rows).to_csv(output_path, index=False)

    print("\nResults saved to:", output_path)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
