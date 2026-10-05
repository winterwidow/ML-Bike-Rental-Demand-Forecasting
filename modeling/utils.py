"""Helpers shared by the notebook and predict.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from modeling.config import DROP_FEATURES, VALID_FRACTION
from preprocessing.data_loader import load_train_data
from preprocessing.features import FEATURE_COLUMNS, make_training_data


def load_xy() -> tuple[pd.DataFrame, pd.Series]:
    """Load train.csv through Naija's shared preprocessing. y is log1p(count)."""
    X, y, warnings = make_training_data(load_train_data())
    for warning in warnings:
        print(f"PREPROCESSING WARNING: {warning}")
    return X, y


def chronological_split(X, y, valid_fraction: float = VALID_FRACTION):
    """Train on the earlier rows, validate on the later rows (no shuffling)."""
    order_key = X["year"] * 1_000_000 + X["month"] * 10_000 + X["day"] * 100 + X["hour"]
    if not order_key.is_monotonic_increasing:
        raise ValueError("Rows are not in time order; sort them before splitting.")
    split = int(len(X) * (1 - valid_fraction))
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


def make_pipeline(model) -> Pipeline:
    """Wrap a model so it drops DROP_FEATURES itself.

    The saved pipeline still accepts all 17 features from make_prediction_features().
    """
    keep = [c for c in FEATURE_COLUMNS if c not in DROP_FEATURES]
    selector = ColumnTransformer(
        [("keep", "passthrough", keep)],
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    return Pipeline([("select", selector), ("model", model)])


def to_counts(pred_log) -> np.ndarray:
    """Convert log1p predictions back to rental counts (never below 0)."""
    return np.clip(np.expm1(pred_log), 0, None)


def score(y_true_log, pred_log) -> dict:
    """RMSLE (main metric), RMSE and R2. RMSE and R2 are on real counts."""
    true_counts, pred_counts = np.expm1(y_true_log), to_counts(pred_log)
    return {
        "RMSLE": round(float(np.sqrt(mean_squared_error(y_true_log, pred_log))), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(true_counts, pred_counts))), 2),
        "R2": round(float(r2_score(true_counts, pred_counts)), 4),
    }


def evaluate(name: str, pipeline: Pipeline, X_valid, y_valid) -> dict:
    """Score a fitted pipeline on the validation set."""
    return {"model": name, **score(y_valid, pipeline.predict(X_valid))}