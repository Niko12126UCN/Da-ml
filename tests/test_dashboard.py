import shutil

import pytest

from api.main import MODELS_DIR, create_app
from ml.store import import_training_runs, log_request
from tests.test_monitoring import LIVE, data, models, store_people  # noqa: F401 (fixtures)
from tests.test_validation import PERSON
from ui import dashboard


def monitoring(models_dir, store):
    return create_app(models_dir, store).test_client().get("/monitoring").get_json()


def test_monitoring_endpoint_on_an_empty_store(models, tmp_path):  # noqa: F811
    body = monitoring(models, tmp_path / "store.db")
    assert body["report"]["status"] == "not enough data" and body["report"]["model"] == LIVE
    assert len(body["daily_requests"]) == 30 and body["model_history"] == []
    assert body["thresholds"] == {"watch": 0.1, "drift": 0.25, "min_predictions": 100}


def test_monitoring_endpoint_with_data(models, tmp_path, data):  # noqa: F811
    store = tmp_path / "store.db"
    store_people(data[1].sample(300, random_state=5).assign(Age=13, BMI=40), models, store)
    client = create_app(models, store).test_client()
    client.post("/predict", json=PERSON)
    client.post("/predict", json={**PERSON, "Age": 99})
    import_training_runs(models, store)
    body = client.get("/monitoring").get_json()
    assert body["report"]["status"] == "drift"
    assert body["daily_requests"][-1] == {"date": body["daily_requests"][-1]["date"], "ok": 1, "invalid": 1}
    assert [m["live"] for m in body["model_history"]] == [True]
    assert "Age" not in str(body["daily_requests"])                      # aggregates only, no answers


def text(component):
    return str(component)


def test_dashboard_renders_every_state(models, tmp_path, data):  # noqa: F811
    empty = monitoring(models, tmp_path / "empty.db")
    page = text(dashboard.body(empty))
    assert "Drift is judged from 100 stored predictions" in page and "No requests in the last 30 days" in page

    store = tmp_path / "drift.db"
    store_people(data[1].sample(300, random_state=6).assign(Age=13, BMI=40), models, store)
    log_request(LIVE, 1, 200, True, store)
    import_training_runs(models, store)
    drifted = monitoring(models, store)
    page = text(dashboard.body(drifted))
    assert "Show as table" in page and "Age group" in page and "Training reference" in page
    assert "Drift" in text(dashboard.status_badge(drifted["report"]["status"], LIVE))

    stable_store = tmp_path / "stable.db"
    store_people(data[1].sample(500, random_state=7), models, stable_store)
    assert monitoring(models, stable_store)["report"]["status"] == "stable"


def test_dashboard_without_reference(tmp_path):
    shutil.copytree(MODELS_DIR / LIVE, tmp_path / "models" / LIVE, ignore=shutil.ignore_patterns("reference.json"))
    (tmp_path / "models" / "latest.txt").write_text(LIVE)
    body = monitoring(tmp_path / "models", tmp_path / "store.db")
    assert "has no reference.json" in text(dashboard.body(body))


def test_model_history_marks_live_and_rejected(models, tmp_path):  # noqa: F811
    body = monitoring(models, tmp_path / "store.db")
    body["model_history"] = [
        {"run_id": "catboost_new", "created_at": "2026-11-18T11:00:00", "data_file": "new.csv", "test_roc_auc": 0.79,
         "test_recall": 0.8, "test_precision": 0.3, "deployed": False, "forced": False,
         "failures": ["test ROC-AUC 0.7900 is below the minimum of 0.8"], "live": False},
        {"run_id": LIVE, "created_at": "2026-11-18T10:04:30", "data_file": "kaggle.csv", "test_roc_auc": 0.8247,
         "test_recall": 0.796, "test_precision": 0.33, "deployed": True, "forced": False, "failures": [],
         "live": True},
    ]
    page = text(dashboard.body(body))
    assert f"● {LIVE}" in page and "no: test ROC-AUC 0.7900 is below the minimum" in page


def test_no_connection(monkeypatch):
    def down(*args, **kwargs):
        raise dashboard.requests.ConnectionError()

    monkeypatch.setattr(dashboard.requests, "get", down)
    from dash import Dash
    test_app = Dash(__name__)
    dashboard.register(test_app, "http://127.0.0.1:1")
    callback = next(iter(test_app.callback_map.values()))["callback"]
    body, status = callback.__wrapped__(0)
    assert "not available" in text(body) and "No connection" in text(status)


@pytest.mark.parametrize("path, marker", [("/", "Diabetes risk check"), ("/dashboard", "Monitoring")])
def test_routing(path, marker):
    from ui.app import show_page
    assert marker in text(show_page(path))
