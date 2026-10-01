"""Generate EDA plots and an evidence-based data-quality report from train.csv."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from preprocessing.data_loader import PROJECT_ROOT, load_train_data
from preprocessing.features import make_features


def _write_report(raw: pd.DataFrame, warnings: list[str], output_path: Path) -> None:
    timestamps = pd.to_datetime(raw["datetime"], errors="coerce")
    target = pd.to_numeric(raw["count"], errors="coerce")
    duplicates = raw.duplicated().sum()
    missing = raw.isna().sum()
    zero_target_share = (target == 0).mean()
    skewness = target.skew()

    measured_issues = []
    if duplicates:
        measured_issues.append(f"- Duplicate rows: {duplicates}; preprocessing removes them.")
    if missing.any():
        measured_issues.append("- Missing values:\n" + "\n".join(
            f"  - `{column}`: {count}" for column, count in missing[missing > 0].items()
        ))
    if zero_target_share > 0.05:
        measured_issues.append(
            f"- Zero-rental share: {zero_target_share:.1%}; use `log1p(count)` so zeros remain valid."
        )
    if abs(skewness) > 1:
        measured_issues.append(
            f"- Target skewness: {skewness:.2f}; evaluate RMSLE and train on `log1p(count)`."
        )
    if timestamps.notna().any() and timestamps.dt.date.nunique() < 30:
        measured_issues.append("- Limited date coverage; validation variance may be high.")
    if not measured_issues:
        measured_issues.append("- No high-severity data-quality issue was detected by the configured checks.")

    report = [
        "# EDA and Preprocessing Report",
        "",
        "## Dataset Summary",
        f"- Rows: {len(raw):,}",
        f"- Columns: {len(raw.columns)}",
        f"- Date range: {timestamps.min()} to {timestamps.max()}",
        f"- Target mean: {target.mean():.2f}",
        f"- Target median: {target.median():.2f}",
        "",
        "## Measured Preprocessing Findings",
        *measured_issues,
        "",
        "## Modeling Risks (Not Accuracy Results)",
        "- `casual` and `registered` are excluded because they sum to `count`, causing target leakage.",
        "- Random train/test splits can overstate performance for time-dependent demand; use chronological validation.",
        "- Low validation accuracy, if observed later, may result from missing calendar/weather interactions or a mismatch between validation dates and future dates. It is not inferred by this EDA script.",
        "",
        "## Actions Taken",
        "- Datetime is parsed into calendar and hour features.",
        "- Invalid numeric values are coerced and median-imputed.",
        "- Duplicate rows and invalid timestamps are removed.",
        "- Temperature buckets and peak-hour features are created.",
    ]
    if warnings:
        report.extend(["", "## Runtime Warnings", *[f"- {warning}" for warning in warnings]])
    output_path.write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    output_dir = PROJECT_ROOT / "outputs" / "eda"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_train_data()
    feature_result = make_features(raw)

    plot_frame = raw.loc[feature_result.features.index].copy()
    plot_frame["hour"] = pd.to_datetime(plot_frame["datetime"]).dt.hour
    plot_frame["day_of_week"] = pd.to_datetime(plot_frame["datetime"]).dt.day_name()
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.barplot(data=plot_frame, x="hour", y="count", estimator="mean", errorbar=None, ax=axes[0, 0])
    axes[0, 0].set_title("Average Rentals by Hour")
    sns.barplot(data=plot_frame, x="day_of_week", y="count", estimator="mean", errorbar=None, ax=axes[0, 1])
    axes[0, 1].set_title("Average Rentals by Weekday")
    axes[0, 1].tick_params(axis="x", rotation=30)
    sns.barplot(data=plot_frame, x="season", y="count", estimator="mean", errorbar=None, ax=axes[1, 0])
    axes[1, 0].set_title("Average Rentals by Season")
    correlations = plot_frame.select_dtypes("number").corr(numeric_only=True)
    sns.heatmap(correlations, cmap="coolwarm", center=0, ax=axes[1, 1])
    axes[1, 1].set_title("Numeric Feature Correlations")
    fig.tight_layout()
    fig.savefig(output_dir / "eda_overview.png", dpi=180)
    plt.close(fig)

    _write_report(raw, feature_result.warnings, output_dir / "eda_report.md")
    print(f"Wrote {output_dir / 'eda_overview.png'}")
    print(f"Wrote {output_dir / 'eda_report.md'}")


if __name__ == "__main__":
    main()
