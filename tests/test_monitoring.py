import json
import shutil

import pandas as pd
import pytest
from catboost import CatBoostClassifier

from api.main import MODELS_DIR, create_app
from ml.monitoring import (MIN_PREDICTIONS, build_reference_for_existing, feature_bins, psi, psi_status, report,
                           volume)
from ml.risk import assess
from ml.store import log_predictions, log_request
from ml.train import DEFAULT_DATA, make_target, split
from ml.validation import FEATURE_COLUMNS, validate_training_data
from tests.test_validation import PERSON

LIVE = (MODELS_DIR / "latest.txt").read_text().strip()


@pytest.fixture(scope="module")
def data():
    df = validate_training_data(pd.read_csv(DEFAULT_DATA))
    X_train, X_test, _, _ = split(df[FEATURE_COLUMNS], make_target(df))
    return X_train, X_test


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    """A copy of the live model with a reference.json built from the Kaggle data."""
    models = tmp_path_factory.mktemp("models")
    shutil.copytree(MODELS_DIR / LIVE, models / LIVE)
    (models / "latest.txt").write_text(LIVE)
    (models / LIVE / "reference.json").unlink(missing_ok=True)
    build_reference_for_existing(LIVE, DEFAULT_DATA, models)
    return models


def store_people(X, models, store):
    model = CatBoostClassifier().load_model(str(models / LIVE / "model.cbm"))
    threshold = json.loads((models / LIVE / "metadata.json").read_text())["threshold"]
    results = [assess(p, threshold) for p in model.predict_proba(X)[:, 1]]
    log_predictions(X, results, LIVE, "ui", store)


def test_psi():
    assert psi([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0)
    assert psi([0.5, 0.5], [0.9, 0.1]) > 0.25
    assert [psi_status(v) for v in (0.05, 0.1, 0.24, 0.25)] == ["stable", "watch", "watch", "drift"]


def test_bins_cover_every_allowed_value(data):
    X_train, _ = data
    for column in FEATURE_COLUMNS:
        bins = feature_bins(column, X_train[column].values)
        covered = [v for lo, hi in bins for v in range(lo, hi + 1)]
        assert covered == sorted(set(covered)), column                       # no overlaps
        assert min(covered) <= X_train[column].min() and max(covered) >= X_train[column].max(), column


def test_reference_file(models):
    reference = json.loads((models / LIVE / "reference.json").read_text())
    assert reference["rows_train"] == 202_944 and reference["rows_test"] == 50_736
    for column in FEATURE_COLUMNS:
        assert sum(reference["features"][column]["shares"]) == pytest.approx(1)
    assert reference["prediction"]["share_flagged"] == pytest.approx(0.380, abs=0.001)   # as in RESULTS.md


def test_not_enough_data(models, tmp_path, data):
    store = tmp_path / "store.db"
    store_people(data[1].head(MIN_PREDICTIONS - 1), models, store)
    r = report(store, models)
    assert r["status"] == "not enough data" and r["stored_predictions"] == MIN_PREDICTIONS - 1


def test_people_like_the_training_data_are_stable(models, tmp_path, data):
    store = tmp_path / "store.db"
    store_people(data[1].sample(2000, random_state=1), models, store)       # unseen people, same population
    r = report(store, models)
    assert r["status"] == "stable", r["features"][:3]
    assert r["prediction"]["share_flagged"] == pytest.approx(0.38, abs=0.04)


def test_a_different_user_group_is_drift(models, tmp_path, data):
    store = tmp_path / "store.db"
    older = data[1].sample(500, random_state=2).assign(Age=13, BMI=40)      # everyone 80+ with BMI 40
    store_people(older, models, store)
    r = report(store, models)
    assert r["status"] == "drift"
    assert {f["feature"] for f in r["features"][:2]} == {"Age", "BMI"}
    assert r["prediction"]["status"] == "drift" and r["prediction"]["mean"] > r["prediction"]["mean_reference"]


def test_model_without_reference(tmp_path):
    shutil.copytree(MODELS_DIR / LIVE, tmp_path / "models" / LIVE, ignore=shutil.ignore_patterns("reference.json"))
    (tmp_path / "models" / "latest.txt").write_text(LIVE)
    assert report(tmp_path / "store.db", tmp_path / "models")["status"] == "no reference"


def test_reference_needs_the_same_data(models, tmp_path):
    other = tmp_path / "other.csv"
    pd.read_csv(DEFAULT_DATA).head(100).to_csv(other, index=False)
    with pytest.raises(ValueError, match="SHA-256"):
        build_reference_for_existing(LIVE, other, models)


def test_volume_and_errors(tmp_path):
    store = tmp_path / "store.db"
    log_request("model_a", 1, 200, True, store)
    log_request("model_a", 3, 200, False, store)
    log_request("model_a", 1, 400, False, store)
    log_request("model_b", 1, 200, False, store)
    v = volume(store, "model_a")
    assert v["requests"] == 3 and v["requests_24h"] == 3 and v["people_predicted"] == 4
    assert v["error_rate"] == pytest.approx(1 / 3) and v["share_stored"] == pytest.approx(1 / 4)


def test_api_counts_every_request_without_answers(tmp_path):
    from ml.store import read_table
    store = tmp_path / "store.db"
    client = create_app(store_path=store).test_client()
    client.post("/predict", json=PERSON)
    client.post("/predict", json={**PERSON, "Age": 99})
    client.post("/predict?store=true", json=[PERSON, PERSON])
    requests = read_table("requests", store)
    assert list(requests["status"]) == [200, 400, 200] and list(requests["n_people"]) == [1, 1, 2]
    assert list(requests["stored"]) == [0, 0, 1]
    assert not set(FEATURE_COLUMNS) & set(requests.columns)                 # no answers in this table
