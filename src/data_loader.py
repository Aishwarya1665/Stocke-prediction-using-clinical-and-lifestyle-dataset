from pathlib import Path
from typing import Optional

import kagglehub
import pandas as pd

DATASET_REF = "fedesoriano/stroke-prediction-dataset"
DATA_FILENAME = "healthcare-dataset-stroke-data.csv"


def load_data(local_dir: Optional[Path] = None) -> pd.DataFrame:
    """Download dataset via kagglehub and return a cleaned DataFrame."""
    download_root = Path(kagglehub.dataset_download(DATASET_REF))
    if local_dir:
        download_root = Path(local_dir)

    file_path = download_root / DATA_FILENAME
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found at {file_path}")

    df = pd.read_csv(file_path)
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    return df
