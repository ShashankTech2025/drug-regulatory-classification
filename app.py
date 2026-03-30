import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "drug_final_pipeline.pkl"

# Features used by the model and form; keep order consistent with training
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


def load_model():
    with MODEL_PATH.open("rb") as f:
        return pickle.load(f)


model = load_model()


def compute_global_feature_importance(model_pipeline):
    """Compute global feature importance for display, if available."""
    clf = getattr(model_pipeline, "named_steps", {}).get("clf")
    if clf is None:
        return []

    importances = None

    # Tree-based models: feature_importances_
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    # Linear models: use absolute value of coefficients
    elif hasattr(clf, "coef_"):
        importances = abs(clf.coef_[0])

    if importances is None:
        return []

    pairs = list(zip(FEATURE_COLUMNS, importances))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs


GLOBAL_FEATURE_IMPORTANCE = compute_global_feature_importance(model)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect form data in the exact order expected by the model
        input_data = {col: request.form.get(col) for col in FEATURE_COLUMNS}

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)

        # Convert numeric columns with safe coercion
        input_df = input_df.apply(pd.to_numeric, errors="coerce")

        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        result = "Regulated Drug" if prediction == 1 else "Non-Regulated Drug"

        # Risk tier based on probability
        prob_percent = probability * 100
        if prob_percent < 33:
            risk_level = "Low risk profile"
            risk_color = "low"
        elif prob_percent < 66:
            risk_level = "Moderate risk profile"
            risk_color = "medium"
        else:
            risk_level = "High risk profile"
            risk_color = "high"

        # Take top 3 global drivers for explanation
        top_features = GLOBAL_FEATURE_IMPORTANCE[:3] if GLOBAL_FEATURE_IMPORTANCE else []

        return render_template(
            "result.html",
            prediction=result,
            probability=round(probability * 100, 2),
            input_values=input_data,
            risk_level=risk_level,
            risk_color=risk_color,
            top_features=top_features,
        )

    except Exception as e:
        return f"Error occurred: {e}"

if __name__ == "__main__":
    app.run(debug=True)