"""Prediction API for the diabetes screening model (step 5 of PROJECT_PLAN.md).

Loads the model named in models/latest.txt and serves:
    GET  /health   -> status and which model is loaded
    GET  /model    -> features, allowed ranges, threshold, risk levels and the model's test metrics
    POST /predict  -> risk for one person (JSON object) or several people (JSON list of objects)

Run from the project root, in the da-ml environment:
    python -m api.main
"""

import json
from pathlib import Path

from catboost import CatBoostClassifier
from flask import Flask, jsonify, request

from ml.risk import DISCLAIMER, HIGH_RISK, assess
from ml.validation import FEATURE_COLUMNS, FEATURE_RANGES, ValidationError, validate_features

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
HOST, PORT = "127.0.0.1", 5001   # same as the taxi project (D-51)
MAX_PEOPLE = 1000                # per request (D-50)


def load_model(models_dir=MODELS_DIR):
    """Load the model named in models_dir/latest.txt. Returns (model, metadata)."""
    run_id = (Path(models_dir) / "latest.txt").read_text(encoding="utf-8").strip()
    run_dir = Path(models_dir) / run_id
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata["features"] != FEATURE_COLUMNS:
        raise RuntimeError(f"model {run_id} was trained on different features than ml/validation.py defines")
    model = CatBoostClassifier().load_model(str(run_dir / metadata["model_file"]))
    return model, metadata


def create_app(models_dir=MODELS_DIR):
    model, metadata = load_model(models_dir)
    threshold = metadata["threshold"]
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify(status="ok", model=metadata["run_id"])

    @app.get("/model")
    def model_info():
        return jsonify(
            model=metadata["run_id"],
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

    @app.post("/predict")
    def predict():
        data = request.get_json(silent=True)
        if data is None:
            return jsonify(errors=["request body must be JSON (Content-Type: application/json)"]), 400
        if isinstance(data, list) and len(data) > MAX_PEOPLE:
            return jsonify(errors=[f"at most {MAX_PEOPLE} people per request, got {len(data)}"]), 400
        try:
            X = validate_features(data)
        except ValidationError as e:
            return jsonify(errors=e.errors), 400

        results = [assess(p, threshold) for p in model.predict_proba(X)[:, 1]]
        body = {"model": metadata["run_id"]}
        if isinstance(data, list):
            body["results"] = results
        else:
            body.update(results[0])
        return jsonify(body)

    return app


if __name__ == "__main__":
    create_app().run(host=HOST, port=PORT, debug=False)
