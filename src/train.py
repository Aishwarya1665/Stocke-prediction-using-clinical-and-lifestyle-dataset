from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import FunctionTransformer

from .data_loader import load_data
from .preprocessing import build_preprocessor, engineer_features
from .utils import save_model


@dataclass
class ModelResult:
    name: str
    auc: float
    best_params: Dict


def model_spaces() -> List[Tuple[str, object, Dict]]:
    return [
        (
            "log_reg",
            LogisticRegression(max_iter=500, n_jobs=-1, class_weight="balanced"),
            {"model__C": [0.5, 1.0, 2.0], "model__penalty": ["l2"], "model__solver": ["lbfgs", "liblinear"]},
        ),
        (
            "random_forest",
            RandomForestClassifier(n_estimators=400, n_jobs=-1, class_weight="balanced"),
            {"model__max_depth": [None, 12, 20], "model__min_samples_leaf": [1, 4]},
        ),
        (
            "grad_boost",
            GradientBoostingClassifier(),
            {"model__n_estimators": [150, 250], "model__learning_rate": [0.05, 0.1], "model__max_depth": [2, 3]},
        ),
    ]


def train_models(df: pd.DataFrame) -> Tuple[Pipeline, List[ModelResult], pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X, y, preprocessor = build_preprocessor(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    best_auc = -1.0
    best_model: Pipeline | None = None
    results: List[ModelResult] = []

    for name, estimator, param_grid in model_spaces():
        pipeline = Pipeline(
            steps=[
                ("features", FunctionTransformer(engineer_features, validate=False)),
                ("preprocess", preprocessor),
                ("smote", SMOTE(random_state=42)),
                ("model", estimator),
            ]
        )

        search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=5,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_train, y_train)

        proba = search.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        results.append(ModelResult(name=name, auc=auc, best_params=search.best_params_))

        if auc > best_auc:
            best_auc = auc
            best_model = search.best_estimator_

    if best_model is None:
        raise RuntimeError("Training failed to produce a model")

    save_model(best_model)
    return best_model, results, X_train, y_train, X_test, y_test
