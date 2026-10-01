"""Feature engineering shared by training, evaluation, and the Streamlit demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data_loader import LEAKAGE_COLUMNS, TARGET_COLUMN

FEATURE_COLUMNS = [
    "season",
    "holiday",
    "workingday",
    "weather",
    "temp",
    "atemp",
    "humidity",
    "windspeed",
    "year",
    "month",
    "day",
    "hour",
    "day_of_week",
    "is_weekend",
    "is_rush_hour",
    "is_weekend_daytime",
    "temperature_bucket",
]


@dataclass(frozen=True)
class FeatureResult:
    """Feature matrix plus warnings discovered while preprocessing."""

    features: pd.DataFrame
    warnings: list[str]


def _coerce_numeric(frame: pd.DataFrame, columns: list[str], warnings: list[str]) -> None:
    for column in columns:
        original_missing = frame[column].isna().sum()
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        new_missing = frame[column].isna().sum() - original_missing
        if new_missing:
            warnings.append(f"{column}: converted {new_missing} non-numeric value(s) to missing.")
        if frame[column].isna().any():
            median = frame[column].median()
            frame[column] = frame[column].fillna(median)
            warnings.append(f"{column}: filled missing values with median {median:.2f}.")


def make_features(raw: pd.DataFrame) -> FeatureResult:
    """Create non-leaking numeric features from raw Bike Sharing Demand records."""
    if "datetime" not in raw:
        raise ValueError("Input data must include a 'datetime' column.")

    warnings: list[str] = []
    frame = raw.copy()
    parsed_datetime = pd.to_datetime(frame["datetime"], errors="coerce")
    bad_datetimes = parsed_datetime.isna().sum()
    if bad_datetimes:
        warnings.append(f"datetime: dropped {bad_datetimes} row(s) that could not be parsed.")
        frame = frame.loc[parsed_datetime.notna()].copy()
        parsed_datetime = parsed_datetime.loc[parsed_datetime.notna()]

    before_duplicates = len(frame)
    frame = frame.drop_duplicates()
    if len(frame) != before_duplicates:
        warnings.append(f"Dropped {before_duplicates - len(frame)} duplicate row(s).")
        parsed_datetime = pd.to_datetime(frame["datetime"], errors="raise")

    numeric_columns = [
        "season", "holiday", "workingday", "weather", "temp", "atemp", "humidity", "windspeed"
    ]
    missing_columns = [column for column in numeric_columns if column not in frame]
    if missing_columns:
        raise ValueError(f"Input data is missing feature columns: {missing_columns}")
    _coerce_numeric(frame, numeric_columns, warnings)

    features = frame[numeric_columns].copy()
    features["year"] = parsed_datetime.dt.year - parsed_datetime.dt.year.min()
    features["month"] = parsed_datetime.dt.month
    features["day"] = parsed_datetime.dt.day
    features["hour"] = parsed_datetime.dt.hour
    features["day_of_week"] = parsed_datetime.dt.dayofweek
    features["is_weekend"] = (features["day_of_week"] >= 5).astype(int)

    weekday_rush = (features["is_weekend"] == 0) & (
        features["hour"].between(7, 9) | features["hour"].between(17, 19)
    )
    features["is_rush_hour"] = weekday_rush.astype(int)
    features["is_weekend_daytime"] = (
        (features["is_weekend"] == 1) & features["hour"].between(10, 18)
    ).astype(int)

    features["temperature_bucket"] = pd.cut(
        features["temp"],
        bins=[-np.inf, 10, 20, 30, np.inf],
        labels=False,
    ).astype(int)

    # Guardrail: target components reveal the answer and must never reach a model.
    accidental_leaks = LEAKAGE_COLUMNS.intersection(features.columns)
    if accidental_leaks:
        raise AssertionError(f"Leakage columns reached the feature matrix: {sorted(accidental_leaks)}")
    return FeatureResult(features=features[FEATURE_COLUMNS], warnings=warnings)


def make_training_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Return X, log1p(count), and preprocessing warnings for model training."""
    if TARGET_COLUMN not in raw:
        raise ValueError("Training data must include 'count'.")
    result = make_features(raw)
    clean_target = pd.to_numeric(raw.loc[result.features.index, TARGET_COLUMN], errors="coerce")
    valid_target = clean_target.notna() & (clean_target >= 0)
    warnings = list(result.warnings)
    if not valid_target.all():
        warnings.append(f"Dropped {(~valid_target).sum()} row(s) with invalid target values.")
    return (
        result.features.loc[valid_target].reset_index(drop=True),
        np.log1p(clean_target.loc[valid_target]).reset_index(drop=True),
        warnings,
    )


def make_prediction_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return the same feature schema for test data or individual app inputs."""
    result = make_features(raw)
    return result.features.reset_index(drop=True), result.warnings
