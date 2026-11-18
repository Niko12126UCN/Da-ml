"""Step 12: the label question, the retraining recommendation and training with user data."""

import json
import sqlite3

import pandas as pd
import pytest

import ml.train
from api.main import create_app
from ml.monitoring import main as monitoring_main, recommendation
from ml.store import connect, labelled_user_rows, log_predictions, read_table
from ml.risk import assess
from ml.validation import FEATURE_COLUMNS, validate_features
from tests.test_monitoring import LIVE, data, models, store_people  # noqa: F401 (fixtures)
from tests.test_train import FAST_PARAMS
from tests.test_validation import PERSON
from ui import dashboard


# --- data store ---

def test_labelled_rows_come_out_in_the_training_format(tmp_path):
    store = tmp_path / "store.db"
    X = validate_features(PERSON)
    log_predictions(X, [assess(0.3, 0.15)], "m", "ui", store, diagnosis=2)
    log_predictions(X, [assess(0.3, 0.15)], "m", "ui", store)                 # no diagnosis given
    rows = labelled_user_rows(store)
    assert len(rows) == 1 and list(rows.columns) == FEATURE_COLUMNS + ["Diabetes_012"]
    assert rows["Diabetes_012"].iloc[0] == 2


def test_old_stores_get_the_new_columns(tmp_path):
    store = tmp_path / "old.db"
    connect(store).close()                                        # today's schema, then remove the new columns
    conn = sqlite3.connect(store)                                 # -> the schema of a store from before D-79/D-80
    conn.execute("ALTER TABLE predictions DROP COLUMN diagnosis")
    conn.execute("ALTER TABLE training_runs DROP COLUMN user_rows")
    conn.commit()
    assert "diagnosis" not in {r[1] for r in conn.execute("PRAGMA table_info(predictions)")}
    conn.close()
    conn = connect(store)
    assert "diagnosis" in {r[1] for r in conn.execute("PRAGMA table_info(predictions)")}
    assert "user_rows" in {r[1] for r in conn.execute("PRAGMA table_info(training_runs)")}
    conn.close()


# --- API ---

def test_diagnosis_is_stored_with_the_answers(tmp_path):
    store = tmp_path / "store.db"
    client = create_app(store_path=store).test_client()
    assert client.post("/predict?store=true&diagnosis=1", json=PERSON).status_code == 200
    assert read_table("predictions", store)["diagnosis"].tolist() == [1]


@pytest.mark.parametrize("query, body", [
    ("?diagnosis=1", PERSON),                      # without storage
    ("?store=true&diagnosis=3", PERSON),           # not 0/1/2
    ("?store=true&diagnosis=yes", PERSON),
    ("?store=true&diagnosis=1", [PERSON, PERSON]),  # only for one person
])
def test_invalid_diagnosis_is_rejected(tmp_path, query, body):
    store = tmp_path / "store.db"
    response = create_app(store_path=store).test_client().post(f"/predict{query}", json=body)
    assert response.status_code == 400 and "diagnosis" in response.get_json()["errors"][0]
    assert read_table("predictions", store).empty


# --- UI ---

@pytest.mark.parametrize("consent, diagnosis, expected", [
    (True, 2, {"store": "true", "source": "ui", "diagnosis": "2"}),
    (True, 0, {"store": "true", "source": "ui", "diagnosis": "0"}),
    (True, "skip", {"store": "true", "source": "ui"}),
    (True, None, {"store": "true", "source": "ui"}),
    (False, 2, {}),                                # never sent without consent
])
def test_ui_sends_the_diagnosis_only_with_consent(monkeypatch, consent, diagnosis, expected):
    import ui.app
    sent = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"probability": 0.33, "risk_level": "high", "message": "m", "disclaimer": "d", "model": "m",
                    "stored": consent}

    def fake_post(url, json, params, timeout):
        sent["params"] = params
        return FakeResponse()

    monkeypatch.setattr(ui.app.requests, "post", fake_post)
    monkeypatch.setattr(ui.app, "cutoffs", lambda model: (0.151, 0.30))
    values = [PERSON[f] for f in ui.app.CHOICE_FIELDS]
    result = str(ui.app.check_risk(1, *values, 175, 96, consent, diagnosis))
    assert sent["params"] == expected
    assert ("meant for people without a diagnosis" in result) == (diagnosis in (1, 2))


# --- recommendation ---

def fake_report(status, features=(), prediction=None):
    return {"status": status, "features": list(features), "prediction": prediction}


def test_recommendation_levels():
    drifted = [{"feature": "Age", "psi": 0.9, "status": "drift"}, {"feature": "Sex", "psi": 0.01, "status": "stable"}]
    rec = recommendation(fake_report("drift", drifted), labelled_rows=42)
    assert rec["level"] == "retrain" and rec["command"] == "python -m ml.train --with-user-data"
    assert len(rec["reasons"]) == 1 and "Age" in rec["reasons"][0] and "42 stored user rows" in rec["data"]
    assert recommendation(fake_report("watch", drifted), 0)["level"] == "watch"
    assert "original dataset only" in recommendation(fake_report("watch", drifted), 0)["data"]
    assert recommendation(fake_report("stable"), 0)["level"] == "none"
    assert recommendation(fake_report("not enough data"), 0)["level"] == "none"


def test_check_exit_code(models, tmp_path, data):  # noqa: F811
    store = tmp_path / "store.db"
    assert monitoring_main(["--store", str(store), "--models", str(models), "--check"]) == 0
    store_people(data[1].sample(300, random_state=11).assign(Age=13, BMI=40), models, store)
    assert monitoring_main(["--store", str(store), "--models", str(models), "--check"]) == 3


def test_dashboard_shows_the_recommendation(models, tmp_path, data):  # noqa: F811
    store = tmp_path / "store.db"
    store_people(data[1].sample(300, random_state=12).assign(Age=13, BMI=40), models, store)
    body = create_app(models, store).test_client().get("/monitoring").get_json()
    page = str(dashboard.body(body))
    assert "Retraining recommended" in page and "python -m ml.train --with-user-data" in page
    empty = create_app(models, tmp_path / "empty.db").test_client().get("/monitoring").get_json()
    assert "Retraining recommended" not in str(dashboard.body(empty))


# --- training with user data ---

def test_training_with_user_data(tmp_path, monkeypatch):
    base = pd.read_csv(ml.train.DEFAULT_DATA).sample(2000, random_state=0)
    base_csv = tmp_path / "base.csv"
    base.to_csv(base_csv, index=False)
    store = tmp_path / "store.db"
    X = validate_features(base.drop(columns="Diabetes_012").head(30).to_dict("records"))
    for diagnosis in [0] * 20 + [2] * 10:
        log_predictions(X.head(1), [assess(0.3, 0.15)], "m", "ui", store, diagnosis=diagnosis)
    log_predictions(X.head(5), [assess(0.3, 0.15)] * 5, "m", "ui", store)       # unlabelled: not used

    seen = {}
    real_train = ml.train.train

    def fast_train(df):
        seen["rows"] = len(df)
        return real_train(df, params=FAST_PARAMS, n_folds=3, log=lambda *_: None)

    monkeypatch.setattr(ml.train, "train", fast_train)
    monkeypatch.setattr(ml.train, "TRAINING_SETS", tmp_path / "training_sets")
    ml.train.main(["--data", str(base_csv), "--out", str(tmp_path / "models"), "--store", str(store),
                   "--with-user-data", "--force"])

    assert seen["rows"] == 2000 + 30
    (run_dir,) = [p for p in (tmp_path / "models").iterdir() if p.is_dir()]
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["rows"]["user_rows"] == 30 and metadata["data"]["file"].startswith("combined_")
    combined = list((tmp_path / "training_sets").glob("combined_*.csv"))
    assert len(combined) == 1 and len(pd.read_csv(combined[0])) == 2030
    assert read_table("training_runs", store)["user_rows"].tolist() == [30]
