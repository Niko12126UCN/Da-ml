"""Train the diabetes screening model and deploy it if it passes the quality check (steps 4 and 8, PROJECT_PLAN.md).

Validates a dataset in the BRFSS 2015 format, trains CatBoost with the settings chosen in Modeling.ipynb, picks the
decision threshold for the recall target, evaluates once on the test split, and saves everything to a new folder
`models/<run_id>/`. If the model passes the quality check (D-61), `models/latest.txt` is switched to it and a running
API picks it up by itself (D-62); otherwise the live model stays and the reasons are printed.

Usage (from the project root, in the da-ml environment):
    python -m ml.train
    python -m ml.train --data path/to/dataset.csv [--out models] [--force]
    python -m ml.train --with-user-data      the dataset + every stored user row with a diagnosis (D-80)
Exit code: 0 = new model is live, 2 = trained but not deployed (quality check failed), 1 = error.
"""

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import catboost
import numpy as np
import pandas as pd
import sklearn
from catboost import CatBoostClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split

from ml.monitoring import build_reference
from ml.risk import HIGH_RISK
from ml.store import DEFAULT_STORE, import_training_runs, labelled_user_rows
from ml.validation import FEATURE_COLUMNS, FEATURE_RANGES, TARGET_COLUMN, validate_training_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "raw_data" / "diabetes_012_health_indicators_BRFSS2015.csv"
DEFAULT_OUT = PROJECT_ROOT / "models"
TRAINING_SETS = PROJECT_ROOT / "data" / "training_sets"   # combined datasets with user rows (not in git, D-80)

SEED = 42                # D-24
TEST_SIZE = 0.2          # D-24
N_FOLDS = 5              # D-24
TARGET_RECALL = 0.80     # D-23
CATBOOST_PARAMS = {"iterations": 500, "learning_rate": 0.1, "depth": 6}   # D-21; ordinal codes as numbers (D-20)

# Quality gate before a new model goes live (D-61)
MIN_ROC_AUC = 0.80        # absolute minimum on the test split
MAX_ROC_AUC_DROP = 0.01   # at most this much below the live model (same margin as D-10)
MIN_RECALL = 0.75         # at the chosen threshold, on the test split (target is 80%, D-23)


def make_target(df):
    """1 = prediabetes or diabetes, 0 = no diabetes (D-08)."""
    return (df[TARGET_COLUMN] > 0).astype(int).rename("diabetes")


def split(X, y):
    """80/20 stratified split with a fixed seed (D-24). Same rows as Modeling.ipynb for the Kaggle file."""
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)


def make_model(params=None):
    return CatBoostClassifier(**(params or CATBOOST_PARAMS), random_seed=SEED, verbose=0, allow_writing_files=False)


def threshold_for_recall(y_true, proba, target_recall=TARGET_RECALL):
    """Highest threshold that still reaches the recall target."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    return float(thresholds[np.where(recall[:-1] >= target_recall)[0].max()])


def evaluate(y_true, proba, threshold):
    """Same metrics as in Modeling.ipynb / RESULTS.md (D-26)."""
    precision, recall, _ = precision_recall_curve(y_true, proba)
    flagged = proba >= threshold
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "precision_at_target_recall": float(precision[recall >= TARGET_RECALL].max()),
        "brier": brier_score_loss(y_true, proba),
        "recall": recall_score(y_true, flagged),
        "precision": precision_score(y_true, flagged),
        "f1": f1_score(y_true, flagged),
        "accuracy": accuracy_score(y_true, flagged),
        "share_flagged": float(flagged.mean()),
    }


def train(df, params=None, n_folds=N_FOLDS, log=print):
    """Validate, split, choose the threshold on out-of-fold predictions, train and evaluate.
    Returns (model, info) where info holds the threshold, metrics and row counts."""
    df = validate_training_data(df)
    X, y = df[FEATURE_COLUMNS], make_target(df)
    X_train, X_test, y_train, y_test = split(X, y)
    log(f"Train: {len(X_train):,} rows ({y_train.mean():.1%} positive), "
        f"test: {len(X_test):,} rows ({y_test.mean():.1%} positive)")

    oof = np.zeros(len(y_train))
    for fold, (tr_idx, va_idx) in enumerate(StratifiedKFold(n_folds, shuffle=True, random_state=SEED)
                                            .split(X_train, y_train), start=1):
        start = time.time()
        fold_model = make_model(params).fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        oof[va_idx] = fold_model.predict_proba(X_train.iloc[va_idx])[:, 1]
        log(f"  CV fold {fold}/{n_folds} done ({time.time() - start:.0f} s)")
    threshold = threshold_for_recall(y_train, oof)
    log(f"Threshold for {TARGET_RECALL:.0%} recall (out-of-fold): {threshold:.4f}")

    start = time.time()
    model = make_model(params).fit(X_train, y_train)
    log(f"Final model trained on {len(X_train):,} rows ({time.time() - start:.0f} s)")

    test_proba = model.predict_proba(X_test)[:, 1]
    info = {
        "threshold": threshold,
        "target_recall": TARGET_RECALL,
        "metrics": {
            "cross_validation": evaluate(y_train, oof, threshold),
            "test": evaluate(y_test, test_proba, threshold),
        },
        # what the training data and the predictions looked like, for monitoring (D-73); saved as reference.json
        "reference": build_reference(X_train, test_proba, threshold, HIGH_RISK),
        "rows": {"total": len(df), "train": len(X_train), "test": len(X_test),
                 "positive_share": float(y.mean())},
        "settings": {"model": "CatBoostClassifier", "params": params or CATBOOST_PARAMS, "seed": SEED,
                     "test_size": TEST_SIZE, "n_folds": n_folds, "balancing": "none"},
    }
    return model, info


def current_metadata(out_dir):
    """Metadata of the model that is live now (named in out_dir/latest.txt), or None if there is none."""
    latest = Path(out_dir) / "latest.txt"
    if not latest.exists():
        return None
    path = Path(out_dir) / latest.read_text(encoding="utf-8").strip() / "metadata.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def quality_check(info, current=None):
    """Quality gate before a new model goes live (D-61). Returns the list of failed checks (empty = passed)."""
    test = info["metrics"]["test"]
    failures = []
    if test["roc_auc"] < MIN_ROC_AUC:
        failures.append(f"test ROC-AUC {test['roc_auc']:.4f} is below the minimum of {MIN_ROC_AUC}")
    if test["recall"] < MIN_RECALL:
        failures.append(f"test recall {test['recall']:.1%} is below the minimum of {MIN_RECALL:.0%}")
    if current is not None:
        current_auc = current["metrics"]["test"]["roc_auc"]
        if test["roc_auc"] < current_auc - MAX_ROC_AUC_DROP:
            failures.append(f"test ROC-AUC {test['roc_auc']:.4f} is more than {MAX_ROC_AUC_DROP} below the live "
                            f"model {current['run_id']} ({current_auc:.4f})")
    return failures


def activate(out_dir, run_id):
    """Make run_id the live model: replace out_dir/latest.txt in one atomic step, so the API (which watches this
    file) never reads a half-written name."""
    latest = Path(out_dir) / "latest.txt"
    tmp = latest.with_suffix(".tmp")
    tmp.write_text(run_id + "\n", encoding="utf-8")
    os.replace(tmp, latest)


def save(model, info, out_dir, data_path, quality=None):
    """Save model + metadata to out_dir/<run_id>/ (does not make it live; see activate). `quality` is the result of
    the quality gate, stored in the metadata."""
    base = datetime.now().strftime("catboost_%Y%m%d_%H%M%S")
    run_id, n = base, 2
    while (Path(out_dir) / run_id).exists():   # two runs in the same second
        run_id, n = f"{base}_{n}", n + 1
    run_dir = Path(out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    model.save_model(str(run_dir / "model.cbm"))

    data_path = Path(data_path)
    metadata = {
        "run_id": run_id,
        "created": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Screening for prediabetes/diabetes risk - not a diagnosis (D-07)",
        "model_file": "model.cbm",
        "features": FEATURE_COLUMNS,
        "feature_ranges": FEATURE_RANGES,
        **{k: v for k, v in info.items() if k != "reference"},
        "data": {"file": data_path.name, "sha256": hashlib.sha256(data_path.read_bytes()).hexdigest()},
        "versions": {"python": platform.python_version(), "catboost": catboost.__version__,
                     "scikit-learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
        "quality_check": quality,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if "reference" in info:
        (run_dir / "reference.json").write_text(json.dumps(info["reference"], indent=1), encoding="utf-8")
    return run_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the diabetes screening model and, if it passes the "
                                                 "quality check, make it live.")
    parser.add_argument("--data", default=DEFAULT_DATA, help="CSV in the BRFSS 2015 Diabetes_012 format")
    parser.add_argument("--out", default=DEFAULT_OUT, help="folder for trained models (default: models/)")
    parser.add_argument("--force", action="store_true", help="make the new model live even if the check fails")
    parser.add_argument("--store", default=DEFAULT_STORE, help="data store where the run is recorded "
                                                               "(default: data/store.db)")
    parser.add_argument("--with-user-data", action="store_true",
                        help="add the stored user rows that have a diagnosis to --data (D-80)")
    args = parser.parse_args(argv)

    print(f"Reading {args.data}")
    df = pd.read_csv(args.data)
    data_path, user_rows = args.data, 0
    if args.with_user_data:
        users = labelled_user_rows(args.store)
        user_rows = len(users)
        df = pd.concat([df, users], ignore_index=True)
        # keep the exact training data (it contains users' answers, so data/training_sets/ is not in git, D-80)
        data_path = TRAINING_SETS / datetime.now().strftime("combined_%Y%m%d_%H%M%S.csv")
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_path, index=False)
        print(f"Added {user_rows:,} labelled user rows from {args.store} -> {len(df):,} rows, saved as {data_path}"
              if user_rows else "No stored user rows have a diagnosis yet; training on the original dataset only.")
    model, info = train(df)
    info["rows"]["user_rows"] = user_rows
    test = info["metrics"]["test"]
    print(f"\nTest set: ROC-AUC {test['roc_auc']:.4f}, recall {test['recall']:.1%}, "
          f"precision {test['precision']:.1%}, flagged {test['share_flagged']:.1%}")

    current = current_metadata(args.out)
    failures = quality_check(info, current)
    go_live = not failures or args.force
    quality = {"passed": not failures, "failures": failures, "forced": bool(failures and args.force),
               "compared_with": current["run_id"] if current else None, "deployed": go_live}
    run_dir = save(model, info, args.out, data_path, quality)
    print(f"Saved to {run_dir}")

    if failures:
        print("\nQuality check FAILED:\n- " + "\n- ".join(failures))
    try:   # record the run (deployed or not) in the data store (D-72); a store problem must not fail training
        import_training_runs(args.out, args.store)
    except Exception as e:
        print(f"Warning: could not record the run in the data store {args.store}: {e}")
    if go_live:
        activate(args.out, run_dir.name)
        print(f"\nNow live: {run_dir.name}" + (" (forced)" if failures else "")
              + ". A running app switches to it automatically.")
        return 0
    print(f"\nNOT deployed. The live model stays {current['run_id'] if current else '(none)'}. "
          "Use --force to deploy anyway.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
