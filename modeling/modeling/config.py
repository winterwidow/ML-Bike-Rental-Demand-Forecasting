"""Shared settings for model training and evaluation."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RESULTS_DIR = OUTPUTS_DIR / "models"
FIGURES_DIR = OUTPUTS_DIR / "figures"

MODEL_PATH = MODELS_DIR / "best_model.joblib"
RESULTS_PATH = RESULTS_DIR / "results.csv"

RANDOM_STATE = 42
VALID_FRACTION = 0.2  # last 20% of rows (in time order) are used for validation

# 'day' is always 1-19 in train.csv and 20-31 in test.csv, so the model can
# never learn anything useful from it. It is dropped inside the model pipeline,
# so the app can still pass all 17 features.
DROP_FEATURES = ["day"]