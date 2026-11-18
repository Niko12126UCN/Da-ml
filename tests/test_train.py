import json
from pathlib import Path

import pandas as pd
import pytest
from catboost import CatBoostClassifier

from ml.train import DEFAULT_DATA, make_target, save, split, train
from ml.validation import FEATURE_COLUMNS, ValidationError, validate_training_data

SPLIT_FILE = Path(__file__).resolve().parents[1] / "data" / "splits" / "train_test_split.csv"
FAST_PARAMS = {"iterations": 20, "learning_rate": 0.3, "depth": 4}


@pytest.fixture(scope="module")
def raw():
    return pd.read_csv(DEFAULT_DATA)


def test_split_matches_shared_split_file(raw):
    df = validate_training_data(raw)
    _, X_test, _, _ = split(df[FEATURE_COLUMNS], make_target(df))
    shared = pd.read_csv(SPLIT_FILE, index_col="row")["split"]
    assert set(X_test.index) == set(shared[shared == "test"].index)


def test_train_and_save_end_to_end(raw, tmp_path):
    sample = raw.sample(5000, random_state=0)
    sample_csv = tmp_path / "sample.csv"
    sample.to_csv(sample_csv, index=False)

    model, info = train(sample, params=FAST_PARAMS, n_folds=3, log=lambda *_: None)
    assert 0 < info["threshold"] < 1
    assert info["metrics"]["cross_validation"]["recall"] >= 0.80     # threshold chosen for 80% recall
    assert info["metrics"]["test"]["roc_auc"] > 0.7
    assert info["rows"] == {"total": 5000, "train": 4000, "test": 1000, "positive_share": pytest.approx(
        (sample["Diabetes_012"] > 0).mean())}

    run_dir = save(model, info, tmp_path / "models", sample_csv)
    assert (tmp_path / "models" / "latest.txt").read_text().strip() == run_dir.name
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["threshold"] == info["threshold"]
    assert metadata["features"] == FEATURE_COLUMNS
    assert len(metadata["data"]["sha256"]) == 64

    loaded = CatBoostClassifier().load_model(str(run_dir / "model.cbm"))
    X = validate_training_data(sample)[FEATURE_COLUMNS]
    assert (loaded.predict_proba(X) == model.predict_proba(X)).all()


def test_invalid_data_is_rejected_before_training(raw):
    bad = raw.head(500).copy()
    bad.loc[0, "Age"] = 99
    with pytest.raises(ValidationError, match="'Age'"):
        train(bad, params=FAST_PARAMS, log=lambda *_: None)
