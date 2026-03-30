## Drug Regulatory Classification ML App

This project is a simple end‑to‑end machine learning application that predicts whether a drug is **Regulated** or **Non‑Regulated** based on a set of numerical features. It includes:

- A **training script** (`train_model.py`) that reads `drug_regulatory_classification_dataset.csv`, trains a scikit‑learn pipeline, evaluates it, and saves the model to `drug_final_pipeline.pkl`.
- A **Flask web app** (`app.py`) with:
  - `index.html` for entering drug features
  - `result.html` for displaying the prediction and probability
  - `static/style.css` for basic styling

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train the model

From the project directory:

```bash
python train_model.py
```

This will create/overwrite `drug_final_pipeline.pkl` in the project root.

### Run the web app

```bash
python app.py
```

Then open `http://127.0.0.1:5000/` in your browser, enter the drug attributes, and submit the form to see the predicted regulatory class and its probability.

