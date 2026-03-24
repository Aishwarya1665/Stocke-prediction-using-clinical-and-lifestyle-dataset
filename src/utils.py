from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def save_model(model: Any, filename: str = "stroke_model.pkl") -> Path:
    path = MODELS_DIR / filename
    joblib.dump(model, path)
    return path


def load_model(path: Path) -> Any:
    return joblib.load(path)


def as_dataframe(records: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([records])
