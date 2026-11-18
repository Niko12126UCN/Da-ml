from pathlib import Path

import pandas as pd
import pytest

from ml.validation import (FEATURE_COLUMNS, TARGET_COLUMN, ValidationError, validate_features,
                           validate_training_data)

RAW_CSV = Path(__file__).resolve().parents[1] / "data" / "raw_data" / "diabetes_012_health_indicators_BRFSS2015.csv"

PERSON = {"HighBP": 1, "HighChol": 1, "CholCheck": 1, "BMI": 31, "Smoker": 0, "Stroke": 0,
          "HeartDiseaseorAttack": 0, "PhysActivity": 1, "Fruits": 1, "Veggies": 1, "HvyAlcoholConsump": 0,
          "AnyHealthcare": 1, "NoDocbcCost": 0, "GenHlth": 3, "MentHlth": 2, "PhysHlth": 5, "DiffWalk": 0,
          "Sex": 1, "Age": 9, "Education": 5, "Income": 6}


def errors_of(func, data):
    with pytest.raises(ValidationError) as info:
        func(data)
    return "\n".join(info.value.errors)


@pytest.fixture(scope="module")
def raw():
    return pd.read_csv(RAW_CSV)


# --- training data ---

def test_real_kaggle_file_passes(raw):
    clean = validate_training_data(raw)
    assert clean.shape == (253_680, 22)
    assert list(clean.columns) == FEATURE_COLUMNS + [TARGET_COLUMN]
    assert all(dtype.kind == "i" for dtype in clean.dtypes)   # floats in the CSV become integers
    pd.testing.assert_frame_equal(clean, raw[clean.columns].astype(int))   # same values, nothing changed


def test_column_order_does_not_matter(raw):
    shuffled = raw[list(reversed(raw.columns))].head(100)
    assert list(validate_training_data(shuffled).columns) == FEATURE_COLUMNS + [TARGET_COLUMN]


def test_binary_kaggle_file_format_is_rejected(raw):
    binary = raw.rename(columns={TARGET_COLUMN: "Diabetes_binary"}).head(100)
    message = errors_of(validate_training_data, binary)
    assert "missing columns: Diabetes_012" in message
    assert "unexpected columns: Diabetes_binary" in message


def test_all_problems_are_reported_at_once(raw):
    bad = raw.head(100).copy()
    bad.loc[0, "Age"] = 14
    bad.loc[1, "GenHlth"] = None
    bad.loc[2, "MentHlth"] = 2.5
    message = errors_of(validate_training_data, bad)
    assert "'Age': 1 value(s) outside 1-13" in message
    assert "'GenHlth': 1 missing value(s)" in message
    assert "'MentHlth': 1 value(s) that are not whole numbers" in message


def test_target_out_of_range_is_rejected(raw):
    bad = raw.head(100).copy()
    bad.loc[0, TARGET_COLUMN] = 3
    assert "'Diabetes_012': 1 value(s) outside 0-2" in errors_of(validate_training_data, bad)


def test_target_with_only_one_class_is_rejected(raw):
    only_healthy = raw[raw[TARGET_COLUMN] == 0].head(100)
    assert "must contain both" in errors_of(validate_training_data, only_healthy)


def test_empty_dataset_is_rejected(raw):
    assert "no rows" in errors_of(validate_training_data, raw.head(0))


# --- prediction input ---

def test_one_person_passes():
    clean = validate_features(PERSON)
    assert clean.shape == (1, 21)
    assert list(clean.columns) == FEATURE_COLUMNS


def test_several_people_pass():
    assert validate_features([PERSON, {**PERSON, "Age": 2}]).shape == (2, 21)


def test_bmi_with_decimals_is_rounded():
    assert validate_features({**PERSON, "BMI": 27.6})["BMI"].iloc[0] == 28


def test_other_decimals_are_rejected():
    assert "'Age': 1 value(s) that are not whole numbers" in errors_of(validate_features, {**PERSON, "Age": 9.5})


def test_whole_number_floats_are_accepted():
    assert validate_features({**PERSON, "Age": 9.0})["Age"].iloc[0] == 9


def test_missing_and_unexpected_fields_are_rejected():
    person = {k: v for k, v in PERSON.items() if k != "Income"}
    person["Weight"] = 80
    message = errors_of(validate_features, person)
    assert "missing columns: Income" in message
    assert "unexpected columns: Weight" in message


def test_target_is_not_allowed_in_prediction_input():
    assert "unexpected columns: Diabetes_012" in errors_of(validate_features, {**PERSON, TARGET_COLUMN: 0})


@pytest.mark.parametrize("value", ["1", None, True, [1]])
def test_non_numbers_are_rejected(value):
    assert "'HighBP' must be a number" in errors_of(validate_features, {**PERSON, "HighBP": value})


@pytest.mark.parametrize("column, value", [("HighBP", 2), ("BMI", 5), ("BMI", 120), ("MentHlth", 31),
                                           ("Age", 0), ("Income", 9)])
def test_out_of_range_values_are_rejected(column, value):
    assert f"'{column}': 1 value(s) outside" in errors_of(validate_features, {**PERSON, column: value})


def test_wrong_types_do_not_hide_other_problems():
    message = errors_of(validate_features, {**PERSON, "BMI": "31", "HighBP": 2, "Age": 14, "Weight": 80})
    assert "'BMI' must be a number" in message
    assert "unexpected columns: Weight" in message


def test_all_value_problems_are_reported_at_once():
    message = errors_of(validate_features, {**PERSON, "BMI": "31", "HighBP": 2, "Age": 14})
    assert "'BMI' must be a number" in message
    assert "'HighBP': 1 value(s) outside 0-1" in message
    assert "'Age': 1 value(s) outside 1-13" in message


@pytest.mark.parametrize("data", [[], "HighBP=1", 42, [PERSON, "not a person"]])
def test_wrong_input_shapes_are_rejected(data):
    with pytest.raises(ValidationError):
        validate_features(data)
