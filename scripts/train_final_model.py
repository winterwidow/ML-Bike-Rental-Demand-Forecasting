"""Rebuild the demo model using the winning XGBoost configuration.

This is intentionally a short, script-based equivalent of the final training
cell in notebooks/02_modeling.ipynb. It is useful when a serialized model file
cannot be loaded on another machine.
"""

from __future__ import annotations

import joblib
from xgboost import XGBRegressor

from modeling.config import MODEL_PATH, MODELS_DIR, RANDOM_STATE
from modeling.utils import chronological_split, evaluate, load_xy, make_pipeline


def build_model():
    """Return the benchmark's winning untuned XGBoost pipeline."""
    return make_pipeline(
        XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=RANDOM_STATE,
        )
    )


def main() -> None:
    X, y = load_xy()
    X_train, X_valid, y_train, y_valid = chronological_split(X, y)

    validation_model = build_model().fit(X_train, y_train)
    print("Chronological validation:", evaluate("XGBoost", validation_model, X_valid, y_valid))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    final_model = build_model().fit(X, y)
    joblib.dump(final_model, MODEL_PATH, compress=3)
    print(f"Saved loadable demo model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
