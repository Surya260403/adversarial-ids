"""
Train and compare a robust Random Forest for the Adversarial IDS project.

Run from the project root:
    python src\robustness_experiment.py

Important:
- Uses train_grouped.csv for fitting and training-derived perturbation statistics.
- Uses test_grouped.csv only for final evaluation.
- Perturbs selected continuous-valued features only; discrete/count/flag/port features remain unchanged.
- This is still a synthetic feature-space stress test, not a realistic packet-level attack.
"""
from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
RANDOM_STATE = 42
N_TREES = 100
MAX_AUGMENTED_ATTACKS = 60_000
PERTURBATION_LEVELS = [0.01, 0.03, 0.05, 0.10]

def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError("Could not find required file: " + " OR ".join(str(p) for p in paths))

def metric_row(model_name, evaluation_name, y_true, y_pred, strength):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "model": model_name,
        "evaluation": evaluation_name,
        "perturbation_strength": strength,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        "attack_detection_rate": cm[1, 1] / max(1, cm[1].sum()),
        "evasion_rate": cm[1, 0] / max(1, cm[1].sum()),
    }

def choose_continuous_features(feature_names):
    # Flow-derived continuous measurements; deliberately exclude ports, counts,
    # TCP flags, window sizes and other discrete/categorical-like fields.
    keywords = (
        "duration", "iat", "bytes/s", "packets/s", "length mean",
        "length std", "length variance", "packet length", "packet size",
        "segment size", "active mean", "active std", "active max",
        "active min", "idle mean", "idle std", "idle max", "idle min",
        "down/up ratio"
    )
    excluded = ("port", "flag", "window", "header length", "subflow", "bulk")
    selected = []
    for name in feature_names:
        low = str(name).strip().lower()
        if any(k in low for k in keywords) and not any(k in low for k in excluded):
            selected.append(name)
    return selected

def main():
    print("=" * 68)
    print("ADVERSARIAL IDS — ROBUST TRAINING & MATCHED EVALUATION")
    print("=" * 68)

    train_path = first_existing([DATA / "train_grouped.csv"])
    test_path = first_existing([DATA / "test_grouped.csv"])
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    feature_path = first_existing([
        MODELS / "grouped_feature_columns.joblib",
        MODELS / "feature_columns.joblib",
    ])
    imputer_path = first_existing([
        MODELS / "grouped_baseline_imputer.joblib",
        MODELS / "baseline_imputer.joblib",
    ])
    baseline_path = first_existing([
        MODELS / "grouped_baseline_random_forest.joblib",
    ])

    feature_columns = list(joblib.load(feature_path))
    imputer = joblib.load(imputer_path)
    baseline_model = joblib.load(baseline_path)

    if "BinaryLabel" not in train_df.columns or "BinaryLabel" not in test_df.columns:
        raise ValueError("Expected a BinaryLabel column in grouped train/test CSVs.")
    missing = [c for c in feature_columns if c not in train_df.columns or c not in test_df.columns]
    if missing:
        raise ValueError(f"Features missing from CSV: {missing[:10]}")

    X_train_raw = train_df[feature_columns].replace([np.inf, -np.inf], np.nan)
    X_test_raw = test_df[feature_columns].replace([np.inf, -np.inf], np.nan)
    y_train = train_df["BinaryLabel"].astype(int).to_numpy()
    y_test = test_df["BinaryLabel"].astype(int).to_numpy()

    print(f"Training rows: {len(train_df):,} | Test rows: {len(test_df):,}")
    print(f"Features: {len(feature_columns)}")
    continuous = choose_continuous_features(feature_columns)
    if not continuous:
        raise ValueError("No continuous features were selected. Inspect feature names before continuing.")
    print(f"Features perturbed: {len(continuous)}")
    print("Perturbed feature examples:", ", ".join(continuous[:8]))

    X_train = imputer.transform(X_train_raw)
    X_test = imputer.transform(X_test_raw)
    # Keep numeric data in float32 where possible to limit memory use.
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    feature_index = {name: i for i, name in enumerate(feature_columns)}
    perturbed_indices = np.array([feature_index[name] for name in continuous], dtype=int)

    # Training-only statistics. No test data contributes to bounds or scales.
    train_cont = X_train[:, perturbed_indices]
    q25 = np.percentile(train_cont, 25, axis=0)
    q75 = np.percentile(train_cont, 75, axis=0)
    iqr = q75 - q25
    # Avoid injecting arbitrary noise into constant-valued features.
    usable = iqr > 1e-12
    perturbed_indices = perturbed_indices[usable]
    iqr = iqr[usable]
    low = np.percentile(X_train[:, perturbed_indices], 1, axis=0)
    high = np.percentile(X_train[:, perturbed_indices], 99, axis=0)
    if len(perturbed_indices) == 0:
        raise ValueError("Selected continuous features have zero IQR in training data.")
    print(f"Non-constant features actually perturbed: {len(perturbed_indices)}")

    def perturb(X, strength, seed):
        out = X.copy()
        rng = np.random.default_rng(seed)
        noise = rng.uniform(-1.0, 1.0, size=(len(out), len(perturbed_indices))).astype(np.float32)
        out[:, perturbed_indices] += noise * (strength * iqr).astype(np.float32)
        out[:, perturbed_indices] = np.clip(out[:, perturbed_indices], low, high)
        return out

    # Augment with perturbed ATTACK training rows only; preserve their ATTACK labels.
    attack_train_idx = np.flatnonzero(y_train == 1)
    rng = np.random.default_rng(RANDOM_STATE)
    n_aug = min(MAX_AUGMENTED_ATTACKS, len(attack_train_idx))
    chosen = rng.choice(attack_train_idx, size=n_aug, replace=False)
    aug_strengths = rng.choice(PERTURBATION_LEVELS, size=n_aug)
    X_aug = np.empty((n_aug, X_train.shape[1]), dtype=np.float32)
    for strength in PERTURBATION_LEVELS:
        positions = np.flatnonzero(aug_strengths == strength)
        if len(positions):
            X_aug[positions] = perturb(X_train[chosen[positions]], strength, RANDOM_STATE + int(strength * 1000))
    y_aug = np.ones(n_aug, dtype=np.int8)
    print(f"Adding {n_aug:,} perturbed ATTACK training examples to robust model.")

    X_robust_train = np.concatenate([X_train, X_aug], axis=0)
    y_robust_train = np.concatenate([y_train, y_aug], axis=0)

    print("Training robust Random Forest (this may take a few minutes)...")
    robust_model = RandomForestClassifier(
        n_estimators=N_TREES,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight=None,
        max_features="sqrt",
    )
    robust_model.fit(X_robust_train, y_robust_train)

    MODELS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    joblib.dump(robust_model, MODELS / "robust_random_forest.joblib", compress=3)
    joblib.dump(imputer, MODELS / "robust_imputer.joblib", compress=3)
    joblib.dump(feature_columns, MODELS / "robust_feature_columns.joblib", compress=3)

    rows = []
    # Clean test set: both models evaluated on exactly the same samples.
    for name, model in [("conventional_rf", baseline_model), ("robust_rf", robust_model)]:
        pred = model.predict(X_test)
        rows.append(metric_row(name, "clean_test", y_test, pred, 0.0))

    # For each stress level, leave BENIGN rows unchanged and perturb ATTACK rows.
    attack_idx = np.flatnonzero(y_test == 1)
    for strength in PERTURBATION_LEVELS:
        X_eval = X_test.copy()
        X_eval[attack_idx] = perturb(X_test[attack_idx], strength, RANDOM_STATE + int(strength * 10000))
        for name, model in [("conventional_rf", baseline_model), ("robust_rf", robust_model)]:
            pred = model.predict(X_eval)
            rows.append(metric_row(name, f"mixed_test_attack_perturb_{int(strength*100)}pct",
                                   y_test, pred, strength))
        # Also save attack-only detection/evasion on the same perturbed attack rows.
        X_attack_eval = X_eval[attack_idx]
        y_attack = y_test[attack_idx]
        for name, model in [("conventional_rf", baseline_model), ("robust_rf", robust_model)]:
            pred_attack = model.predict(X_attack_eval)
            rows.append(metric_row(name, f"attack_only_perturb_{int(strength*100)}pct",
                                   y_attack, pred_attack, strength))

    metrics = pd.DataFrame(rows)
    metrics_path = RESULTS / "robustness_comparison.csv"
    metrics.to_csv(metrics_path, index=False)

    # Persist a concise record of exactly which feature dimensions were perturbed.
    pd.DataFrame({
        "feature": [feature_columns[i] for i in perturbed_indices],
        "training_iqr": iqr,
        "training_p01": low,
        "training_p99": high,
    }).to_csv(RESULTS / "perturbed_feature_bounds.csv", index=False)

    print("\nRESULTS — CLEAN TEST")
    print(metrics[metrics["evaluation"] == "clean_test"][
        ["model", "accuracy", "precision", "recall", "f1_score", "tn", "fp", "fn", "tp"]
    ].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nRESULTS — ATTACK-ONLY SYNTHETIC STRESS TEST")
    attack_rows = metrics[metrics["evaluation"].str.startswith("attack_only")]
    print(attack_rows[["model", "evaluation", "attack_detection_rate", "evasion_rate", "fn"]]
          .to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print(f"\nSaved robust model: {MODELS / 'robust_random_forest.joblib'}")
    print(f"Saved comparison:   {metrics_path}")
    print(f"Saved feature bounds: {RESULTS / 'perturbed_feature_bounds.csv'}")
    print("\nDone. Results are synthetic feature-space robustness measurements, not proof of real-world evasion.")

if __name__ == "__main__":
    main()
