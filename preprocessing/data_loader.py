"""Load and validate the Bike Sharing Demand CSV files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

BASE_COLUMNS = {
    "datetime",
    "season",
    "holiday",
    "workingday",
    "weather",
    "temp",
    "atemp",
    "humidity",
    "windspeed",
}
TARGET_COLUMN = "count"
LEAKAGE_COLUMNS = {"casual", "registered"}


def _resolve_csv(filename: str) -> Path:
    path = RAW_DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Put the Kaggle-style CSV file in data/raw/ "
            "and keep its name as train.csv or test.csv."
        )
    return path


def load_csv(filename: str, require_target: bool) -> pd.DataFrame:
    """Load one CSV and validate its minimum schema."""
    frame = pd.read_csv(_resolve_csv(filename))
    missing = BASE_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{filename} is missing required columns: {sorted(missing)}")
    if require_target and TARGET_COLUMN not in frame.columns:
        raise ValueError(f"{filename} must include the target column '{TARGET_COLUMN}'.")
    return frame


def load_train_data() -> pd.DataFrame:
    """Load the labelled training data."""
    return load_csv("train.csv", require_target=True)


def load_test_data() -> pd.DataFrame:
    """Load the unlabelled test data."""
    return load_csv("test.csv", require_target=False)
