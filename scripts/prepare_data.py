"""Create model-ready CSV files without using a notebook."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from preprocessing.data_loader import PROJECT_ROOT, load_test_data, load_train_data
from preprocessing.features import make_prediction_features, make_training_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Create processed bike-rental features.")
    parser.add_argument("--include-test", action="store_true", help="Also process data/raw/test.csv.")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_x, train_y, warnings = make_training_data(load_train_data())
    pd.concat([train_x, train_y.rename("log1p_count")], axis=1).to_csv(
        output_dir / "train_features.csv", index=False
    )

    if args.include_test:
        test_x, test_warnings = make_prediction_features(load_test_data())
        test_x.to_csv(output_dir / "test_features.csv", index=False)
        warnings.extend(test_warnings)

    print(f"Wrote processed files to {output_dir}")
    for warning in warnings:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
