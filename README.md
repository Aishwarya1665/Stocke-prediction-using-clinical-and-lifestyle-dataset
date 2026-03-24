# Stroke Prediction using Clinical & Lifestyle Data

End-to-end pipeline for predicting stroke risk with clinical and lifestyle attributes. The project covers ingestion, preprocessing, modeling, evaluation, API, and a Streamlit front-end.

## Project Layout

- data/ — dataset download target (cached by kagglehub)
- notebooks/ — exploratory notebooks
- src/ — core logic (ingestion, preprocessing, training, evaluation)
- models/ — persisted models
- api/ — FastAPI service
- app/ — Streamlit UI
- main.py — runnable training pipeline

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Authenticate Kaggle locally (kagglehub reuses your Kaggle credentials).

## Running Training

```bash
python main.py
```

This downloads data via kagglehub, preprocesses, performs model selection with SMOTE and cross-validation, evaluates, and saves the best pipeline to models/stroke_model.pkl.

## API

```bash
uvicorn api.app:app --reload
```

POST /predict example:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "age": 55,
    "hypertension": 0,
    "heart_disease": 0,
    "ever_married": "Yes",
    "work_type": "Private",
    "residence_type": "Urban",
    "avg_glucose_level": 105.5,
    "bmi": 29.1,
    "smoking_status": "never smoked"
  }'
```

## Streamlit App

```bash
streamlit run app/streamlit_app.py
```

## Notes

- Handles missing BMI, encodes categorical features, adds age/bmi groupings, scales numerics.
- Models tried: Logistic Regression, Random Forest, Gradient Boosting. Selection via ROC-AUC.
- Class imbalance addressed with SMOTE and class weights where applicable.
