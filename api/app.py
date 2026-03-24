from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.utils import load_model, MODELS_DIR

MODEL_PATH = MODELS_DIR / "stroke_model.pkl"

app = FastAPI(title="Stroke Prediction API")


class PatientFeatures(BaseModel):
    gender: Literal["Male", "Female", "Other"]
    age: float = Field(..., ge=0)
    hypertension: int = Field(..., ge=0, le=1)
    heart_disease: int = Field(..., ge=0, le=1)
    ever_married: Literal["Yes", "No"]
    work_type: Literal["children", "Govt_job", "Never_worked", "Private", "Self-employed"]
    residence_type: Literal["Urban", "Rural"]
    avg_glucose_level: float = Field(..., ge=0)
    bmi: float = Field(..., ge=0)
    smoking_status: Literal["formerly smoked", "never smoked", "smokes", "Unknown"]


def load_pipeline():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model not found. Run training first.")
    return load_model(MODEL_PATH)


@app.get("/")
async def root():
    return {
        "message": "Stroke Prediction API",
        "docs": "/docs",
        "predict_endpoint": "POST /predict",
    }


@app.post("/predict")
async def predict(features: PatientFeatures):
    try:
        model = load_pipeline()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    df = pd.DataFrame([features.model_dump()])
    proba = model.predict_proba(df)[:, 1][0]
    label = int(proba >= 0.5)
    return {"prediction": label, "probability": proba}
