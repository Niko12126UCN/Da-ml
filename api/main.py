"""Prediction API for the diabetes screening model (step 5 of PROJECT_PLAN.md).

Serves the model named in models/latest.txt:
    GET  /health   -> status and which model is loaded
    GET  /model    -> features, allowed ranges, threshold, risk levels and the model's test metrics
    GET  /monitoring -> drift report, requests per day and model history for the dashboard (aggregates only)
    POST /predict  -> risk for one person (JSON object) or several people (JSON list of objects)
                      add ?store=true to store the answers and results anonymously in the data store (opt-in, D-68),
                      and ?source=ui when the request comes from the web UI

When training switches models/latest.txt to a new model, the API loads it and switches over by itself, without a
restart (D-62). If the new model can't be loaded, the current one keeps serving and the error is logged.

Run from the project root, in the da-ml environment:
    python -m api.main
"""

import json
import logging
import threading
from pathlib import Path

from catboost import CatBoostClassifier
from flask import Flask, jsonify, request

from ml.monitoring import dashboard_data
from ml.risk import DISCLAIMER, HIGH_RISK, assess
from ml.store import DEFAULT_STORE, log_predictions, log_request
from ml.validation import FEATURE_COLUMNS, FEATURE_RANGES, ValidationError, validate_features

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
HOST, PORT = "127.0.0.1", 5001   # same as the taxi project (D-51)
MAX_PEOPLE = 1000                # per request (D-50)


log = logging.getLogger("api")


def live_run_id(models_dir=MODELS_DIR):
    return (Path(models_dir) / "latest.txt").read_text(encoding="utf-8").strip()


def load_model(models_dir=MODELS_DIR, run_id=None):
    """Load a model (default: the one named in models_dir/latest.txt). Returns (model, metadata)."""
    run_id = run_id or live_run_id(models_dir)
    run_dir = Path(models_dir) / run_id
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata["features"] != FEATURE_COLUMNS:
        raise RuntimeError(f"model {run_id} was trained on different features than ml/validation.py defines")
    model = CatBoostClassifier().load_model(str(run_dir / metadata["model_file"]))
    return model, metadata


class ModelStore:
    """Holds the live model and switches to a new one when models/latest.txt changes (D-62)."""

    def __init__(self, models_dir=MODELS_DIR):
        self.models_dir = Path(models_dir)
        self._lock = threading.Lock()
        self._current = load_model(self.models_dir)   # fails loudly at start-up if there is no usable model
        self._failed = None                            # run ID that failed to load (not retried every request)

    def get(self):
        """(model, metadata) of the live model, switching first if latest.txt names a different one."""
        try:
            wanted = live_run_id(self.models_dir)
        except OSError:
            return self._current
        if wanted in (self._current[1]["run_id"], self._failed):
            return self._current
        with self._lock:
            if wanted != self._current[1]["run_id"]:
                try:
                    self._current = load_model(self.models_dir, wanted)
                    log.warning("Switched to model %s", wanted)
                except Exception as e:   # keep serving the current model; a broken new model must not stop the API
                    self._failed = wanted
                    log.error("Could not load model %s, keeping %s: %s", wanted, self._current[1]["run_id"], e)
        return self._current


def create_app(models_dir=MODELS_DIR, store_path=DEFAULT_STORE):
    store = ModelStore(models_dir)
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify(status="ok", model=store.get()[1]["run_id"])

    @app.get("/model")
    def model_info():
        metadata = store.get()[1]
        threshold = metadata["threshold"]
        return jsonify(
            model=metadata["run_id"],
            created=metadata.get("created"),
            purpose=metadata["purpose"],
            features=FEATURE_COLUMNS,
            feature_ranges=FEATURE_RANGES,
            risk_levels={"low": f"below {threshold:.4f}", "elevated": f"{threshold:.4f} to {HIGH_RISK}",
                         "high": f"{HIGH_RISK} and above"},
            threshold=threshold,
            high_risk=HIGH_RISK,
            test_metrics=metadata["metrics"]["test"],
            disclaimer=DISCLAIMER,
        )

    @app.get("/monitoring")
    def monitoring():
        """Monitoring report, requests per day and model history for the dashboard (D-77). Aggregates only."""
        store.get()   # switch first if a new model went live, so the report is about the live model
        return jsonify(dashboard_data(store_path, models_dir))

    @app.post("/predict")
    def predict():
        data = request.get_json(silent=True)
        body, status = handle_predict(data)
        n_people = len(data) if isinstance(data, list) else int(isinstance(data, dict))
        try:   # count every request for monitoring (D-75); never stops the response (D-71)
            log_request(body.get("model") or store.get()[1]["run_id"], n_people, status, body.get("stored", False),
                        store_path)
        except Exception:
            log.exception("Could not count the request in %s", store_path)
        return jsonify(body), status

    def handle_predict(data):
        """Returns (response body, HTTP status)."""
        if data is None:
            return {"errors": ["request body must be JSON (Content-Type: application/json)"]}, 400
        if isinstance(data, list) and len(data) > MAX_PEOPLE:
            return {"errors": [f"at most {MAX_PEOPLE} people per request, got {len(data)}"]}, 400
        try:
            X = validate_features(data)
        except ValidationError as e:
            return {"errors": e.errors}, 400

        model, metadata = store.get()
        results = [assess(p, metadata["threshold"]) for p in model.predict_proba(X)[:, 1]]
        body = {"model": metadata["run_id"], "stored": False}
        if request.args.get("store") == "true":
            source = "ui" if request.args.get("source") == "ui" else "api"
            try:
                log_predictions(X, results, metadata["run_id"], source, store_path)
                body["stored"] = True
            except Exception:   # the data store must never stop a prediction (D-71)
                log.exception("Could not store the prediction in %s", store_path)
        if isinstance(data, list):
            body["results"] = results
        else:
            body.update(results[0])
        return body, 200

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    create_app().run(host=HOST, port=PORT, debug=False)
