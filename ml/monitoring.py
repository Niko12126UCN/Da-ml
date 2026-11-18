"""Data and model monitoring (step 10 of PROJECT_PLAN.md).

Compares what real users answered (the stored predictions, opt-in) with what the live model was trained on:
- data drift: per question, the Population Stability Index (PSI) between the training rows and the stored answers,
- prediction drift: PSI of the predicted probabilities (reference: the model's test-set predictions), plus the share
  flagged, the mean risk and the share per risk level,
- volume and errors: all /predict requests (counts only, no answers), from the store's `requests` table.

The reference for each model is saved at training time as models/<run_id>/reference.json (D-73).
True outcomes never come back from users (D-66), so live accuracy can't be monitored.

Usage (from the project root, in the da-ml environment):
    python -m ml.monitoring                                    report for the live model
    python -m ml.monitoring --build-reference <run_id> --data <csv>   create reference.json for an existing model
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ml.store import DEFAULT_STORE, read_table
from ml.validation import FEATURE_COLUMNS, FEATURE_RANGES

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

PSI_WATCH = 0.10          # PSI from here: "watch"   (D-74)
PSI_DRIFT = 0.25          # PSI from here: "drift"   (D-74)
MIN_PREDICTIONS = 100     # fewer stored predictions of the live model: "not enough data" (D-74)
EPS = 1e-4                # replaces empty shares in the PSI formula
DAY_BINS = [(0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, 29), (30, 30)]   # MentHlth / PhysHlth
STATUS_ORDER = ["not enough data", "stable", "watch", "drift"]


# --- reference (saved with every model) ---

def feature_bins(column, values):
    """Bins as inclusive (low, high) value ranges. Codes: one bin per allowed value. BMI: deciles of the training
    data. Days: fixed groups."""
    if column == "BMI":
        edges = np.unique(np.quantile(values, np.linspace(0, 1, 11)).round())
        lo_all, hi_all = FEATURE_RANGES["BMI"]
        edges[0], edges[-1] = lo_all, hi_all
        return [(int(edges[i]) if i == 0 else int(edges[i]) + 1, int(edges[i + 1])) for i in range(len(edges) - 1)]
    if column in ("MentHlth", "PhysHlth"):
        return DAY_BINS
    lo, hi = FEATURE_RANGES[column]
    return [(v, v) for v in range(lo, hi + 1)]


def shares(values, bins):
    values = np.asarray(values)
    counts = np.array([((values >= lo) & (values <= hi)).sum() for lo, hi in bins], dtype=float)
    return (counts / max(len(values), 1)).tolist()


def probability_edges(proba):
    """Decile edges of the reference probabilities (first/last opened up to 0 and 1)."""
    edges = np.unique(np.quantile(proba, np.linspace(0, 1, 11)))
    edges[0], edges[-1] = 0.0, 1.0 + 1e-9
    return edges.tolist()


def probability_shares(proba, edges):
    counts, _ = np.histogram(np.asarray(proba, dtype=float), bins=edges)
    return (counts / max(len(proba), 1)).tolist()


def level_shares(proba, threshold, high_risk):
    proba = np.asarray(proba, dtype=float)
    return {"low": float((proba < threshold).mean()),
            "elevated": float(((proba >= threshold) & (proba < high_risk)).mean()),
            "high": float((proba >= high_risk).mean())}


def build_reference(X_train, test_proba, threshold, high_risk):
    """What the model's world looked like: answer shares on the training rows, predictions on the test rows."""
    features = {}
    for column in FEATURE_COLUMNS:
        bins = feature_bins(column, X_train[column].values)
        features[column] = {"bins": bins, "shares": shares(X_train[column].values, bins)}
    edges = probability_edges(test_proba)
    return {
        "rows_train": len(X_train),
        "rows_test": len(test_proba),
        "features": features,
        "prediction": {"edges": edges, "shares": probability_shares(test_proba, edges),
                       "mean": float(np.mean(test_proba)), "share_flagged": float(np.mean(test_proba >= threshold)),
                       "levels": level_shares(test_proba, threshold, high_risk)},
    }


# --- comparison ---

def psi(expected, actual):
    """Population Stability Index: sum((actual - expected) * ln(actual / expected)) over the bins."""
    e = np.clip(np.asarray(expected, dtype=float), EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_status(value):
    if value >= PSI_DRIFT:
        return "drift"
    if value >= PSI_WATCH:
        return "watch"
    return "stable"


def load_reference(run_id, models_dir=MODELS_DIR):
    path = Path(models_dir) / run_id / "reference.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def live_model(models_dir=MODELS_DIR):
    run_id = (Path(models_dir) / "latest.txt").read_text(encoding="utf-8").strip()
    metadata = json.loads((Path(models_dir) / run_id / "metadata.json").read_text(encoding="utf-8"))
    return run_id, metadata


def volume(store_path=DEFAULT_STORE, model=None, now=None):
    """Request counts (all /predict calls, no answers): total, last 24 h / 7 days, errors, share stored."""
    requests = read_table("requests", store_path)
    if model is not None:
        requests = requests[requests["model"] == model]
    now = now or datetime.now(timezone.utc)
    times = pd.to_datetime(requests["created_at"], utc=True)
    ok = requests["status"] == 200
    people = requests.loc[ok, "n_people"]
    return {
        "requests": int(len(requests)),
        "requests_24h": int((times >= now - timedelta(days=1)).sum()),
        "requests_7d": int((times >= now - timedelta(days=7)).sum()),
        "people_predicted": int(people.sum()),
        "error_rate": float((~ok).mean()) if len(requests) else 0.0,
        "share_stored": float(requests.loc[ok, "stored"].mul(requests.loc[ok, "n_people"]).sum() / people.sum())
        if people.sum() else 0.0,
    }


def report(store_path=DEFAULT_STORE, models_dir=MODELS_DIR):
    """Monitoring report for the live model. All predictions it made that were stored count (D-74)."""
    from ml.risk import HIGH_RISK
    run_id, metadata = live_model(models_dir)
    reference = load_reference(run_id, models_dir)
    predictions = read_table("predictions", store_path)
    current = predictions[predictions["model"] == run_id]
    result = {"model": run_id, "stored_predictions": int(len(current)), "min_predictions": MIN_PREDICTIONS,
              "volume": volume(store_path, run_id), "reference_available": reference is not None,
              "features": [], "prediction": None}
    if reference is None:
        result["status"] = "no reference"
        return result
    if len(current) < MIN_PREDICTIONS:
        result["status"] = "not enough data"
        return result

    for column in FEATURE_COLUMNS:
        ref = reference["features"][column]
        value = psi(ref["shares"], shares(current[column].values, ref["bins"]))
        result["features"].append({"feature": column, "psi": value, "status": psi_status(value)})
    result["features"].sort(key=lambda f: f["psi"], reverse=True)

    ref_pred = reference["prediction"]
    proba = current["probability"].values
    pred_psi = psi(ref_pred["shares"], probability_shares(proba, ref_pred["edges"]))
    result["prediction"] = {
        "psi": pred_psi, "status": psi_status(pred_psi),
        "mean": float(proba.mean()), "mean_reference": ref_pred["mean"],
        "share_flagged": float(current["flagged"].mean()), "share_flagged_reference": ref_pred["share_flagged"],
        "levels": level_shares(proba, metadata["threshold"], HIGH_RISK), "levels_reference": ref_pred["levels"],
    }
    statuses = [f["status"] for f in result["features"]] + [result["prediction"]["status"]]
    result["status"] = max(statuses, key=STATUS_ORDER.index)
    return result


# --- reference for existing models ---

def build_reference_for_existing(run_id, data_path, models_dir=MODELS_DIR):
    """Create reference.json for a model trained before references existed, from the same data and split."""
    from catboost import CatBoostClassifier
    from ml.risk import HIGH_RISK
    from ml.train import make_target, split
    from ml.validation import validate_training_data

    run_dir = Path(models_dir) / run_id
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    sha = hashlib.sha256(Path(data_path).read_bytes()).hexdigest()
    if sha != metadata["data"]["sha256"]:
        raise ValueError(f"{data_path} is not the file {run_id} was trained on (SHA-256 differs)")
    df = validate_training_data(pd.read_csv(data_path))
    X_train, X_test, _, _ = split(df[FEATURE_COLUMNS], make_target(df))
    model = CatBoostClassifier().load_model(str(run_dir / metadata["model_file"]))
    reference = build_reference(X_train, model.predict_proba(X_test)[:, 1], metadata["threshold"], HIGH_RISK)
    (run_dir / "reference.json").write_text(json.dumps(reference, indent=1), encoding="utf-8")
    return run_dir / "reference.json"


def print_report(r):
    print(f"Live model: {r['model']}")
    v = r["volume"]
    print(f"Requests: {v['requests']} ({v['requests_24h']} in the last 24 h, {v['requests_7d']} in 7 days), "
          f"{v['people_predicted']} people predicted, error rate {v['error_rate']:.1%}, "
          f"stored (opt-in) {v['share_stored']:.0%}")
    print(f"Stored predictions of this model: {r['stored_predictions']} (minimum for drift: {r['min_predictions']})")
    print(f"\nSTATUS: {r['status'].upper()}")
    if r["prediction"]:
        p = r["prediction"]
        print(f"\nPredictions: PSI {p['psi']:.3f} ({p['status']}); mean risk {p['mean']:.1%} "
              f"(reference {p['mean_reference']:.1%}); flagged {p['share_flagged']:.1%} "
              f"(reference {p['share_flagged_reference']:.1%})")
        print("\nQuestions with the largest shift (PSI):")
        for f in r["features"][:8]:
            print(f"  {f['feature']:22s} {f['psi']:.3f}  {f['status']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Monitoring report for the live model.")
    parser.add_argument("--store", default=DEFAULT_STORE, help="data store (default: data/store.db)")
    parser.add_argument("--models", default=MODELS_DIR, help="models folder (default: models/)")
    parser.add_argument("--build-reference", metavar="RUN_ID", help="create reference.json for an existing model")
    parser.add_argument("--data", help="the CSV the model was trained on (with --build-reference)")
    args = parser.parse_args(argv)
    if args.build_reference:
        print(f"Saved {build_reference_for_existing(args.build_reference, args.data, args.models)}")
        return 0
    print_report(report(args.store, args.models))
    return 0


if __name__ == "__main__":
    sys.exit(main())
