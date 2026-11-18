import json
import shutil

import pytest

from api.main import MODELS_DIR
from ml.risk import assess
from ml.store import connect, import_training_runs, log_predictions, read_table
from ml.validation import FEATURE_COLUMNS, validate_features
from tests.test_validation import PERSON


@pytest.fixture
def store(tmp_path):
    return tmp_path / "store.db"


def test_tables_are_created(store):
    conn = connect(store)
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert {"predictions", "training_runs"} <= tables


def test_predictions_are_stored_without_identifying_columns(store):
    X = validate_features([PERSON, {**PERSON, "Age": 2}])
    results = [assess(0.33, 0.151), assess(0.05, 0.151)]
    assert log_predictions(X, results, "model_a", "ui", store) == 2

    df = read_table("predictions", store)
    assert len(df) == 2
    assert list(df["risk_level"]) == ["high", "low"] and list(df["flagged"]) == [1, 0]
    assert list(df["Age"]) == [PERSON["Age"], 2] and (df["model"] == "model_a").all()
    assert set(df.columns) == {"id", "created_at", "model", "probability", "risk_level", "flagged", "source",
                               *FEATURE_COLUMNS, "diagnosis"}        # nothing like name, IP or browser (D-67)
    assert df["diagnosis"].isna().all()                              # optional label, not given here
    assert df["created_at"].str.endswith("+00:00").all()             # UTC


def test_training_runs_are_imported_once(store, tmp_path):
    models = tmp_path / "models"
    shutil.copytree(MODELS_DIR, models, ignore=shutil.ignore_patterns("autogluon_benchmark"))
    n_models = len(list(models.glob("*/metadata.json")))
    assert import_training_runs(models, store) == n_models
    assert import_training_runs(models, store) == 0                   # idempotent
    runs = read_table("training_runs", store)
    assert len(runs) == n_models and runs["test_roc_auc"].gt(0.8).all()


def test_rejected_runs_are_recorded_as_not_deployed(store, tmp_path):
    run = tmp_path / "models" / "catboost_rejected"
    run.mkdir(parents=True)
    (run / "metadata.json").write_text(json.dumps({
        "run_id": "catboost_rejected", "created": "2026-11-18T10:00:00", "threshold": 0.15,
        "metrics": {"test": {"roc_auc": 0.7, "recall": 0.8, "precision": 0.3}},
        "quality_check": {"deployed": False, "forced": False, "failures": ["test ROC-AUC too low"]}}))
    import_training_runs(tmp_path / "models", store)
    row = read_table("training_runs", store).iloc[0]
    assert row["deployed"] == 0 and json.loads(row["failures"]) == ["test ROC-AUC too low"]


def test_unknown_table_is_refused(store):
    with pytest.raises(ValueError):
        read_table("sqlite_master; DROP TABLE predictions", store)
