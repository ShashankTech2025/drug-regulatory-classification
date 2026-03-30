import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("drug_regulatory_classification_dataset.csv")
MODEL_PATH = Path("drug_final_pipeline.pkl")

# Features used in the web form and model
FEATURE_COLUMNS = [
    "Dosage_mg",
    "Price_Per_Unit",
    "Production_Cost",
    "Marketing_Spend",
    "Clinical_Trial_Phase",
    "Side_Effect_Severity_Score",
    "Abuse_Potential_Score",
    "Prescription_Rate",
]

TARGET_COLUMN = "Target_Regulatory_Class"
TARGET_MAPPING = {
    "Non-Regulated Drug": 0,
    "Regulated Drug": 1,
}


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Keep only the columns we actually use in the app
    missing_cols = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns in dataset: {missing_cols}")
    return df


def build_pipelines() -> dict[str, Pipeline]:
    """Build multiple candidate pipelines with different classifiers."""
    pipelines: dict[str, Pipeline] = {}

    # 1. Logistic Regression (baseline)
    pipelines["logreg"] = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    solver="lbfgs",
                ),
            ),
        ]
    )

    # 2. Random Forest (handles non‑linearities, feature interactions)
    pipelines["rf"] = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=None,
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )

    # 3. Gradient Boosting (strong tabular baseline)
    pipelines["gb"] = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                GradientBoostingClassifier(
                    n_estimators=150,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=42,
                ),
            ),
        ]
    )

    return pipelines


def main() -> None:
    print("Loading data...")
    df = load_data(DATA_PATH)

    # Drop rows without a target label
    df = df.dropna(subset=[TARGET_COLUMN])

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].map(TARGET_MAPPING)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipelines = build_pipelines()

    best_name: str | None = None
    best_pipeline: Pipeline | None = None
    best_acc: float = -1.0

    print("Training and evaluating candidate models...")
    for name, pipe in pipelines.items():
        print(f"\n=== Training {name} ===")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"{name} accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_pipeline = pipe

    if best_pipeline is None or best_name is None:
        raise RuntimeError("No best model found during training.")

    print(f"\nBest model: {best_name} with accuracy {best_acc:.4f}")
    print("Classification report for best model:")
    y_pred_best = best_pipeline.predict(X_test)
    print(classification_report(y_test, y_pred_best, target_names=list(TARGET_MAPPING.keys())))

    print(f"Saving best pipeline ({best_name}) to {MODEL_PATH}...")
    with MODEL_PATH.open("wb") as f:
        pickle.dump(best_pipeline, f)

    print("Done.")


if __name__ == "__main__":
    main()

