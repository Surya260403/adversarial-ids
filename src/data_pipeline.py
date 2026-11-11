from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = (
    PROJECT_ROOT / "data" / "MachineLearningCSV" / "MachineLearningCVE"
)
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"


def find_csv_files():
    """Find all CIC-IDS2017 CSV files in the raw data directory."""
    return sorted(RAW_DATA_DIR.glob("*.csv"))


def clean_column_names(df):
    """Remove leading/trailing whitespace from column names."""
    df.columns = df.columns.str.strip()
    return df


def clean_labels(df):
    """Normalize the original CIC-IDS2017 labels."""
    df["Label"] = df["Label"].astype(str).str.strip()
    return df


def create_binary_label(df):
    """Create a binary target: BENIGN=0, all attacks=1."""
    df["BinaryLabel"] = (df["Label"] != "BENIGN").astype(int)
    return df


def clean_numeric_values(df):
    """Replace infinite values with missing values."""
    numeric_columns = df.select_dtypes(include="number").columns

    df[numeric_columns] = df[numeric_columns].replace(
        [float("inf"), float("-inf")],
        pd.NA,
    )

    return df


def remove_duplicates(df):
    """Remove exact duplicate rows."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)

    return df, removed


def load_csv(path):
    """Load one CSV file and perform basic cleaning."""
    df = pd.read_csv(path)

    df = clean_column_names(df)
    df = clean_labels(df)
    df = create_binary_label(df)
    df = clean_numeric_values(df)

    df, duplicates_removed = remove_duplicates(df)

    return df, duplicates_removed


def load_all_csv_files():
    """Load and combine all CIC-IDS2017 CSV files."""
    csv_files = find_csv_files()

    dataframes = []

    print(f"Found {len(csv_files)} CSV files.")

    for path in csv_files:
        print(f"Loading: {path.name}")

        df, duplicates_removed = load_csv(path)

        print(f"  Rows after cleaning: {len(df)}")
        print(f"  Duplicates removed: {duplicates_removed}")

        dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)

    return combined_df


def show_label_distribution(df):
    """Display original and binary label distributions."""
    print("\nOriginal label distribution:")
    print(df["Label"].value_counts())

    print("\nBinary label distribution:")
    print(df["BinaryLabel"].value_counts().sort_index())


def create_balanced_dataset(
    df,
    benign_count=250_000,
    attack_count=250_000,
):
    """Create a balanced dataset with equal benign and attack samples."""

    benign_df = df[df["BinaryLabel"] == 0]
    attack_df = df[df["BinaryLabel"] == 1]

    if len(benign_df) < benign_count:
        raise ValueError("Not enough benign samples.")

    if len(attack_df) < attack_count:
        raise ValueError("Not enough attack samples.")

    benign_sample = benign_df.sample(
        n=benign_count,
        random_state=42,
    )

    # Preserve rare attack classes.
    rare_attack_mask = (
        attack_df["Label"].str.contains(
            "Heartbleed|Sql Injection|Infiltration",
            case=False,
            regex=True,
            na=False,
        )
    )

    rare_attack_df = attack_df[rare_attack_mask]
    remaining_attack_df = attack_df[~rare_attack_mask]

    remaining_count = attack_count - len(rare_attack_df)

    remaining_sample = remaining_attack_df.sample(
        n=remaining_count,
        random_state=42,
    )

    attack_sample = pd.concat(
        [rare_attack_df, remaining_sample],
        ignore_index=True,
    )

    balanced_df = pd.concat(
        [benign_sample, attack_sample],
        ignore_index=True,
    )

    balanced_df = balanced_df.sample(
        frac=1,
        random_state=42,
    ).reset_index(drop=True)

    return balanced_df


def split_dataset(df):
    """Split dataset into stratified training and testing sets."""

    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["BinaryLabel"],
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, test_df


def prepare_features(train_df, test_df):
    """
    Separate features and target and perform leakage-safe
    missing-value imputation.
    """

    # Remove non-feature columns.
    feature_columns = [
        column
        for column in train_df.columns
        if column not in ["Label", "BinaryLabel"]
    ]

    X_train = train_df[feature_columns].copy()
    X_test = test_df[feature_columns].copy()

    y_train = train_df["BinaryLabel"].copy()
    y_test = test_df["BinaryLabel"].copy()

    # Make sure all feature columns are numeric.
    X_train = X_train.apply(pd.to_numeric, errors="coerce")
    X_test = X_test.apply(pd.to_numeric, errors="coerce")

    # Learn imputation values ONLY from training data.
    imputer = SimpleImputer(strategy="median")

    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    return X_train, X_test, y_train, y_test, feature_columns, imputer


def train_random_forest(X_train, y_train):
    """Train the baseline Random Forest classifier."""

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the IDS model using classification metrics."""

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    print("\n========== BASELINE RANDOM FOREST ==========")

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["BENIGN", "ATTACK"],
            zero_division=0,
        )
    )

    return predictions


def save_model(model, imputer, feature_columns):
    """Save the trained model and preprocessing information."""

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "baseline_random_forest.joblib"
    imputer_path = MODEL_DIR / "baseline_imputer.joblib"
    features_path = MODEL_DIR / "feature_columns.joblib"

    import joblib

    joblib.dump(model, model_path)
    joblib.dump(imputer, imputer_path)
    joblib.dump(feature_columns, features_path)

    print("\nSaved model:")
    print(model_path)

    print("Saved imputer:")
    print(imputer_path)

    print("Saved feature list:")
    print(features_path)


def save_results(y_test, predictions):
    """Save baseline evaluation metrics."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "f1_score": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
    }

    results_df = pd.DataFrame([results])

    results_path = RESULTS_DIR / "baseline_random_forest_metrics.csv"

    results_df.to_csv(results_path, index=False)

    print("\nSaved metrics:")
    print(results_path)


def main():
    """Build dataset, train baseline RF, and evaluate it."""

    # 1. Load and clean all raw files.
    df = load_all_csv_files()

    print("\nCombined dataset shape:", df.shape)

    # 2. Inspect original distribution.
    show_label_distribution(df)

    # 3. Create balanced 500k dataset.
    balanced_df = create_balanced_dataset(df)

    print("\nBalanced dataset shape:", balanced_df.shape)

    print("\nBalanced binary distribution:")
    print(
        balanced_df["BinaryLabel"]
        .value_counts()
        .sort_index()
    )

    print("\nBalanced attack distribution:")
    print(
        balanced_df[
            balanced_df["BinaryLabel"] == 1
        ]["Label"].value_counts()
    )

    # 4. Save balanced dataset.
    balanced_path = PROJECT_ROOT / "data" / "balanced_dataset.csv"

    balanced_df.to_csv(
        balanced_path,
        index=False,
    )

    print("\nSaved balanced dataset:")
    print(balanced_path)

    # 5. Train/test split.
    train_df, test_df = split_dataset(balanced_df)

    print("\nTrain shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    print("\nTrain binary distribution:")
    print(train_df["BinaryLabel"].value_counts().sort_index())

    print("\nTest binary distribution:")
    print(test_df["BinaryLabel"].value_counts().sort_index())

    # 6. Save train/test datasets.
    train_path = PROJECT_ROOT / "data" / "train.csv"
    test_path = PROJECT_ROOT / "data" / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\nSaved train dataset:")
    print(train_path)

    print("Saved test dataset:")
    print(test_path)

    # 7. Prepare ML features.
    X_train, X_test, y_train, y_test, feature_columns, imputer = (
        prepare_features(train_df, test_df)
    )

    print("\nFeature matrix:")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("Number of features:", len(feature_columns))

    # 8. Train baseline Random Forest.
    print("\nTraining baseline Random Forest...")

    model = train_random_forest(
        X_train,
        y_train,
    )

    print("Random Forest training complete.")

    # 9. Evaluate baseline.
    predictions = evaluate_model(
        model,
        X_test,
        y_test,
    )

    # 10. Save model and preprocessing.
    save_model(
        model,
        imputer,
        feature_columns,
    )

    # 11. Save metrics.
    save_results(
        y_test,
        predictions,
    )


if __name__ == "__main__":
    main()