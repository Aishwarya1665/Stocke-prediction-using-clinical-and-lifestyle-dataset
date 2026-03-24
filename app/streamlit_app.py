import streamlit as st
import pandas as pd

from src.utils import load_model, MODELS_DIR

MODEL_PATH = MODELS_DIR / "stroke_model.pkl"


def load_pipeline():
    return load_model(MODEL_PATH)


def main():
    st.title("Stroke Risk Prediction")

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        age = st.slider("Age", min_value=1, max_value=100, value=45)
        hypertension = st.selectbox("Hypertension", [0, 1])
        heart_disease = st.selectbox("Heart Disease", [0, 1])
        ever_married = st.selectbox("Ever Married", ["Yes", "No"])
    with col2:
        work_type = st.selectbox("Work Type", ["children", "Govt_job", "Never_worked", "Private", "Self-employed"])
        residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
        avg_glucose_level = st.slider("Average Glucose", min_value=40.0, max_value=300.0, value=110.0)
        bmi = st.slider("BMI", min_value=10.0, max_value=60.0, value=27.5)
        smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

    if st.button("Predict"):
        if not MODEL_PATH.exists():
            st.error("Model not found. Run training first.")
            return

        model = load_pipeline()
        features = pd.DataFrame([
            {
                "gender": gender,
                "age": age,
                "hypertension": hypertension,
                "heart_disease": heart_disease,
                "ever_married": ever_married,
                "work_type": work_type,
                "residence_type": residence_type,
                "avg_glucose_level": avg_glucose_level,
                "bmi": bmi,
                "smoking_status": smoking_status,
            }
        ])
        proba = model.predict_proba(features)[:, 1][0]
        label = "High Risk" if proba >= 0.5 else "Low Risk"

        st.metric(label="Risk", value=label, delta=f"Probability {proba:.2%}")


if __name__ == "__main__":
    main()
