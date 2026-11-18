"""Input validation shared by the training script and the API (D-14).

Defines the "exact format" of the BRFSS 2015 diabetes dataset (`diabetes_012_health_indicators_BRFSS2015.csv`)
and checks data against it. Bad input is rejected with a list of *all* problems found, never silently used.

- `validate_training_data(df)`  -> a whole dataset: 21 features + `Diabetes_012`
- `validate_features(data)`     -> input for a prediction: 21 features, one or more people

Both return a clean integer DataFrame with the columns in the canonical order (FEATURE_COLUMNS).
"""

import numbers

import numpy as np
import pandas as pd

TARGET_COLUMN = "Diabetes_012"

# Allowed (min, max) per feature, all whole numbers. Codes from the BRFSS 2015 codebook (see EDA.ipynb section 3).
BINARY = (0, 1)
FEATURE_RANGES = {
    "HighBP": BINARY,
    "HighChol": BINARY,
    "CholCheck": BINARY,
    "BMI": (12, 98),        # range observed in the dataset (D-38)
    "Smoker": BINARY,
    "Stroke": BINARY,
    "HeartDiseaseorAttack": BINARY,
    "PhysActivity": BINARY,
    "Fruits": BINARY,
    "Veggies": BINARY,
    "HvyAlcoholConsump": BINARY,
    "AnyHealthcare": BINARY,
    "NoDocbcCost": BINARY,
    "GenHlth": (1, 5),      # 1 = excellent ... 5 = poor
    "MentHlth": (0, 30),    # days in the past 30 days
    "PhysHlth": (0, 30),    # days in the past 30 days
    "DiffWalk": BINARY,
    "Sex": BINARY,          # 0 = female, 1 = male
    "Age": (1, 13),         # 1 = 18-24 ... 13 = 80+
    "Education": (1, 6),    # 1 = never attended school ... 6 = college graduate
    "Income": (1, 8),       # 1 = < $10,000 ... 8 = $75,000+
}
FEATURE_COLUMNS = list(FEATURE_RANGES)  # canonical order, same as in the raw CSV
TARGET_RANGE = (0, 2)                   # 0 = no diabetes, 1 = prediabetes, 2 = diabetes

# BMI may be sent with decimals (e.g. calculated from height and weight); it is rounded like in the dataset (D-38).
ROUNDED_COLUMNS = {"BMI"}


class ValidationError(ValueError):
    """Raised when data doesn't match the expected format. `errors` lists every problem found."""

    def __init__(self, errors):
        self.errors = list(errors)
        super().__init__("Invalid input:\n- " + "\n- ".join(self.errors))


def validate_training_data(df):
    """Validate a whole dataset (21 features + Diabetes_012). Returns a clean integer DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise ValidationError([f"expected a pandas DataFrame, got {type(df).__name__}"])
    clean = _validate_frame(df, {**FEATURE_RANGES, TARGET_COLUMN: TARGET_RANGE})
    if clean[TARGET_COLUMN].eq(0).all() or clean[TARGET_COLUMN].gt(0).all():
        raise ValidationError([f"'{TARGET_COLUMN}' must contain both people without (0) and with (1/2) diabetes"])
    return clean


def validate_features(data):
    """Validate input for a prediction: one person (dict), several people (list of dicts) or a DataFrame,
    with exactly the 21 feature columns. Returns a clean integer DataFrame."""
    if isinstance(data, dict):
        records = [data]
    elif isinstance(data, list):
        records = data
    elif isinstance(data, pd.DataFrame):
        return _validate_frame(data, FEATURE_RANGES)
    else:
        raise ValidationError([f"expected a dict, a list of dicts or a DataFrame, got {type(data).__name__}"])

    if not records:
        raise ValidationError(["no input given"])
    errors, rows = [], []
    for i, record in enumerate(records):
        if not isinstance(record, dict):
            errors.append(f"row {i}: expected an object with feature values, got {type(record).__name__}")
            continue
        row = dict(record)
        for column, value in record.items():
            # JSON values: only real numbers are allowed (no strings, no true/false, no null)
            if column in FEATURE_RANGES and (isinstance(value, bool) or not isinstance(value, numbers.Real)):
                errors.append(f"row {i}: '{column}' must be a number, got {value!r}")
                row[column] = FEATURE_RANGES[column][0]  # placeholder, so the other checks still run
        rows.append(row)
    if rows:
        try:
            clean = _validate_frame(pd.DataFrame(rows), FEATURE_RANGES)
        except ValidationError as e:
            errors += e.errors
    if errors:
        raise ValidationError(errors)
    return clean


def _validate_frame(df, ranges):
    expected = list(ranges)
    missing = [c for c in expected if c not in df.columns]
    unexpected = [c for c in df.columns if c not in ranges]
    errors = []
    if missing:
        errors.append(f"missing columns: {', '.join(missing)}")
    if unexpected:
        errors.append(f"unexpected columns: {', '.join(map(str, unexpected))}")
    if df.columns.duplicated().any():
        errors.append(f"duplicate columns: {', '.join(df.columns[df.columns.duplicated()])}")
    if errors:
        raise ValidationError(errors)
    if df.empty:
        raise ValidationError(["no rows"])

    clean = {}
    for column in expected:
        values = pd.to_numeric(df[column], errors="coerce")
        problems = []
        n_missing = int(df[column].isna().sum())
        n_not_numeric = int((values.isna() & df[column].notna()).sum())
        if n_missing:
            problems.append(f"{n_missing} missing value(s)")
        if n_not_numeric:
            example = df[column][values.isna() & df[column].notna()].iloc[0]
            problems.append(f"{n_not_numeric} non-numeric value(s), e.g. {example!r}")
        values = values.dropna()
        if column in ROUNDED_COLUMNS:
            values = values.round()
        not_whole = values[values != np.floor(values)]
        if len(not_whole):
            problems.append(f"{len(not_whole)} value(s) that are not whole numbers, e.g. {not_whole.iloc[0]}")
        lo, hi = ranges[column]
        out_of_range = values[(values < lo) | (values > hi)]
        if len(out_of_range):
            problems.append(f"{len(out_of_range)} value(s) outside {lo}-{hi}, e.g. {out_of_range.iloc[0]:g}")
        if problems:
            errors.append(f"'{column}': " + "; ".join(problems))
        else:
            clean[column] = values.astype(int)

    if errors:
        raise ValidationError(errors)
    return pd.DataFrame(clean, index=df.index)[expected]
