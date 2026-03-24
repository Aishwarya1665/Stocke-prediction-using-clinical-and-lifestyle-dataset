from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

TARGET_COL = "stroke"
ID_COL = "id"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    if ID_COL in data.columns:
        data = data.drop(columns=[ID_COL])

    if "bmi" in data.columns:
        median_bmi = data["bmi"].median()
        data["bmi"] = data["bmi"].fillna(median_bmi)

    if "age" in data.columns:
        bins = [0, 25, 45, 65, np.inf]
        labels = ["young", "adult", "midlife", "senior"]
        data["age_group"] = pd.cut(data["age"], bins=bins, labels=labels, right=False)

    if "bmi" in data.columns:
        bmi_bins = [0, 18.5, 25, 30, np.inf]
        bmi_labels = ["underweight", "healthy", "overweight", "obese"]
        data["bmi_category"] = pd.cut(data["bmi"], bins=bmi_bins, labels=bmi_labels, right=False)

    return data


def build_preprocessor(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    if TARGET_COL not in df.columns:
        raise ValueError(f"'{TARGET_COL}' column is required")

    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    sample = engineer_features(X)
    categorical_cols: List[str] = sample.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_cols: List[str] = sample.select_dtypes(include=["number"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )

    return X, y, preprocessor
