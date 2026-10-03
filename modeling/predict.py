"""Prediction helpers used by the Streamlit demo.

Typical use in the app:
    from modeling.predict import predict_one
    count = predict_one("2012-06-15 08:00", temp=24, humidity=55, windspeed=10)
"""

from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from modeling.config import MODEL_PATH
from preprocessing.features import make_prediction_features

# In train.csv the first year (2011) is coded as year = 0 and 2012 as 1.
TRAIN_START_YEAR = 2011


@lru_cache(maxsize=1)
def load_model():
    """Load the saved model pipeline once and reuse it."""
    if not MODEL_PATH.exists():
           raise FileNotFoundError(
       f"Model not found at {MODEL_PATH}. Pull the latest branch, or create it with "
       "`python -m scripts.train_final_model`."
   )
    return joblib.load(MODEL_PATH)


def predict_count(raw: pd.DataFrame) -> np.ndarray:
    """Predict hourly rental counts for raw rows (same columns as test.csv).

    Needs: datetime, season, holiday, workingday, weather, temp, atemp,
    humidity, windspeed.
    """
    features, _ = make_prediction_features(raw)  # Naija's shared preprocessing
    if len(features) != len(raw):
        raise ValueError(
            "Some rows were dropped during preprocessing (bad datetime or "
            "duplicate rows). Fix the input and try again."
        )

    # Naija's code measures 'year' from the smallest year in the input, which is
    # wrong for a single row. Re-measure it from the training start year.
    parsed = pd.to_datetime(raw["datetime"]).reset_index(drop=True)
    features["year"] = parsed.dt.year - TRAIN_START_YEAR

    pred_log = load_model().predict(features)         # model predicts log1p(count)
    counts = np.clip(np.expm1(pred_log), 0, None)     # back to real counts, never < 0
    return np.round(counts).astype(int)


def predict_one(when, temp, humidity, windspeed, weather=1, holiday=0, atemp=None) -> int:
    """Predict rentals for one hour.

    when:      date and time, e.g. "2012-06-15 08:00" or a datetime
    temp:      temperature in Celsius
    humidity:  0-100
    windspeed: same units as train.csv
    weather:   1 clear, 2 mist, 3 light rain/snow, 4 heavy rain/snow
    holiday:   1 if it is a holiday, else 0
    atemp:     'feels like' temperature; defaults to temp if not given
    """
    when = pd.Timestamp(when)
    row = pd.DataFrame([{
        "datetime": when,
        "season": (when.month - 1) // 3 + 1,                    # Jan-Mar = 1 ... Oct-Dec = 4
        "holiday": int(holiday),
        "workingday": int(when.dayofweek < 5 and not holiday),  # Mon-Fri and not a holiday
        "weather": int(weather),
        "temp": float(temp),
        "atemp": float(temp if atemp is None else atemp),
        "humidity": float(humidity),
        "windspeed": float(windspeed),
    }])
    return int(predict_count(row)[0])


if __name__ == "__main__":
    # Quick demo: python -m modeling.predict
    print("Summer weekday 8 am :", predict_one("2012-06-15 08:00", 24, 55, 10))
    print("Summer weekday 3 am :", predict_one("2012-06-15 03:00", 20, 70, 5))
    print("Winter Sunday noon  :", predict_one("2012-01-15 12:00", 5, 60, 8))