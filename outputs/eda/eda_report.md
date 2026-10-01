# EDA and Preprocessing Report

## Dataset Summary
- Rows: 10,886
- Columns: 12
- Date range: 2011-01-01 00:00:00 to 2012-12-19 23:00:00
- Target mean: 191.57
- Target median: 145.00

## Measured Preprocessing Findings
- Target skewness: 1.24; evaluate RMSLE and train on `log1p(count)`.

## Modeling Risks (Not Accuracy Results)
- `casual` and `registered` are excluded because they sum to `count`, causing target leakage.
- Random train/test splits can overstate performance for time-dependent demand; use chronological validation.
- Low validation accuracy, if observed later, may result from missing calendar/weather interactions or a mismatch between validation dates and future dates. It is not inferred by this EDA script.

## Actions Taken
- Datetime is parsed into calendar and hour features.
- Invalid numeric values are coerced and median-imputed.
- Duplicate rows and invalid timestamps are removed.
- Temperature buckets and peak-hour features are created.
