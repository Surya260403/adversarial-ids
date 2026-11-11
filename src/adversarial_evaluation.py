from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "test.csv"

MODEL_PATH = PROJECT_ROOT / "models" / "baseline_random_forest.joblib"
IMPUTER_PATH = PROJECT_ROOT / "models" / "baseline_imputer.joblib"
FEATURES_PATH = PROJECT_ROOT / "models" / "feature_columns.joblib"

RESULTS_DIR = PROJECT_ROOT / "results"


# Reproducibility
RANDOM_STATE = 42

# Percentage perturbation applied to numerical features.
# Multiple strengths allow us to measure degradation.
PERTURBATION_LEVELS = [0.01, 0.03, 0.05, 0.10]


def load_artifacts():
    """Load the trained IDS model and preprocessing artifacts."""

    model = joblib.load(MODEL_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    feature_columns = joblib.load(FEATURES_PATH)

    return model, imputer, feature_columns


def load_test_data(feature_columns):
    """Load the test dataset and separate features and target."""

    df = pd.read_csv(DATA_PATH)

    X = df[feature_columns].copy()
    y = df["BinaryLabel"].copy()

    X = X.apply(pd.to_numeric, errors="coerce")

    return df, X, y


def prepare_features(X, imputer):
    """Apply the same training-time imputer to test features."""

    return imputer.transform(X)


def select_attack_samples(X, y):
    """Select only attack samples for evasion evaluation."""

    attack_mask = y.to_numpy() == 1

    X_attack = X[attack_mask]
    y_attack = y.iloc[np.where(attack_mask)[0]].reset_index(drop=True)

    return X_attack, y_attack


def create_feature_bounds(X_train_like):
    """
    Create robust feature bounds from the test attack data.

    Bounds use the 1st and 99th percentiles to prevent extreme
    synthetic values during perturbation.
    """

    lower_bounds = np.nanpercentile(
        X_train_like,
        1,
        axis=0,
    )

    upper_bounds = np.nanpercentile(
        X_train_like,
        99,
        axis=0,
    )

    return lower_bounds, upper_bounds


def perturb_features(
    X,
    perturbation_strength,
    lower_bounds,
    upper_bounds,
    random_state=42,
):
    """
    Apply controlled feature-space perturbations to all features.

    Each feature is perturbed proportionally to its magnitude.
    Values are clipped to robust feature bounds.
    """

    rng = np.random.default_rng(random_state)

    scale = np.maximum(np.abs(X), 1.0)

    direction = rng.choice(
        [-1.0, 1.0],
        size=X.shape,
    )

    noise = (
        direction
        * perturbation_strength
        * scale
    )

    X_perturbed = X + noise

    X_perturbed = np.clip(
        X_perturbed,
        lower_bounds,
        upper_bounds,
    )

    return X_perturbed


def calculate_metrics(y_true, predictions):
    """Calculate adversarial evaluation metrics."""

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

    true_benign = matrix[0, 0]
    false_attack = matrix[0, 1]
    false_benign = matrix[1, 0]
    true_attack = matrix[1, 1]

    total_attack = false_benign + true_attack

    if total_attack > 0:
        attack_detection_rate = true_attack / total_attack
        evasion_rate = false_benign / total_attack
    else:
        attack_detection_rate = 0.0
        evasion_rate = 0.0

    total = matrix.sum()

    accuracy = (
        (true_benign + true_attack) / total
        if total > 0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "attack_detection_rate": attack_detection_rate,
        "evasion_rate": evasion_rate,
        "true_benign": true_benign,
        "false_attack": false_attack,
        "false_benign": false_benign,
        "true_attack": true_attack,
    }


def run_adversarial_evaluation(
    model,
    X_attack,
    y_attack,
    lower_bounds,
    upper_bounds,
):
    """Evaluate the IDS at multiple perturbation strengths."""

    results = []

    # Baseline attack-only evaluation.
    baseline_predictions = model.predict(X_attack)

    baseline_metrics = calculate_metrics(
        y_attack,
        baseline_predictions,
    )

    baseline_metrics["perturbation_strength"] = 0.0
    baseline_metrics["experiment"] = "baseline"

    results.append(baseline_metrics)

    print("\n========== ADVERSARIAL EVALUATION ==========")

    print("\nBaseline attack evaluation:")
    print(
        f"Attack detection rate: "
        f"{baseline_metrics['attack_detection_rate']:.4f}"
    )

    print(
        f"Evasion rate: "
        f"{baseline_metrics['evasion_rate']:.4f}"
    )

    for strength in PERTURBATION_LEVELS:

        print(
            f"\nTesting perturbation strength: "
            f"{strength:.0%}"
        )

        X_adversarial = perturb_features(
            X_attack,
            perturbation_strength=strength,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            random_state=RANDOM_STATE,
        )

        predictions = model.predict(
            X_adversarial
        )

        metrics = calculate_metrics(
            y_attack,
            predictions,
        )

        metrics["perturbation_strength"] = strength
        metrics["experiment"] = "adversarial"

        results.append(metrics)

        print(
            f"Attack detection rate: "
            f"{metrics['attack_detection_rate']:.4f}"
        )

        print(
            f"Evasion rate: "
            f"{metrics['evasion_rate']:.4f}"
        )

        print(
            f"Attack → BENIGN: "
            f"{metrics['false_benign']}"
        )

    return pd.DataFrame(results)


def main():
    """Run controlled adversarial evaluation against the baseline IDS."""

    print("Loading trained model...")

    model, imputer, feature_columns = load_artifacts()

    print(
        f"Number of model features: "
        f"{len(feature_columns)}"
    )

    print("\nLoading test dataset...")

    df, X, y = load_test_data(feature_columns)

    print("Test dataset shape:", df.shape)

    X = prepare_features(
        X,
        imputer,
    )

    # Evaluate only malicious flows.
    X_attack, y_attack = select_attack_samples(
        X,
        y,
    )

    print(
        "Attack samples evaluated:",
        len(X_attack),
    )

    print(
        "Features perturbed:",
        X_attack.shape[1],
    )

    # Build robust bounds from the attack feature space.
    lower_bounds, upper_bounds = create_feature_bounds(
        X_attack
    )

    results_df = run_adversarial_evaluation(
        model=model,
        X_attack=X_attack,
        y_attack=y_attack,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        RESULTS_DIR
        / "adversarial_evaluation.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print("\n========== FINAL RESULTS ==========")
    print(results_df)

    print("\nSaved results:")
    print(output_path)


if __name__ == "__main__":
    main()