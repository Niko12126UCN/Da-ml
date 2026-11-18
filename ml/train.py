"""Train the diabetes screening model (step 4 of PROJECT_PLAN.md).

Validates a dataset in the BRFSS 2015 format, trains CatBoost with the settings chosen in Modeling.ipynb, picks the
decision threshold for the recall target, evaluates once on the test split, and saves everything to a new folder
`models/<run_id>/`. `models/latest.txt` then names that folder, so the API loads the newest model.

Usage (from the project root, in the da-ml environment):
    python -m ml.train
    python -m ml.train --data path/to/dataset.csv --out models
"""

import argparse
import hashlib
import json
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

from ml.validation import FEATURE_COLUMNS, FEATURE_RANGES, TARGET_COLUMN, validate_training_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "raw_data" / "diabetes_012_health_indicators_BRFSS2015.csv"
DEFAULT_OUT = PROJECT_ROOT / "models"

SEED = 42                # D-24
TEST_SIZE = 0.2          # D-24
N_FOLDS = 5              # D-24
TARGET_RECALL = 0.80     # D-23
CATBOOST_PARAMS = {"iterations": 500, "learning_rate": 0.1, "depth": 6}   # D-21; ordinal codes as numbers (D-20)


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

    info = {
        "threshold": threshold,
        "target_recall": TARGET_RECALL,
        "metrics": {
            "cross_validation": evaluate(y_train, oof, threshold),
            "test": evaluate(y_test, model.predict_proba(X_test)[:, 1], threshold),
        },
        "rows": {"total": len(df), "train": len(X_train), "test": len(X_test),
                 "positive_share": float(y.mean())},
        "settings": {"model": "CatBoostClassifier", "params": params or CATBOOST_PARAMS, "seed": SEED,
                     "test_size": TEST_SIZE, "n_folds": n_folds, "balancing": "none"},
    }
    return model, info


def save(model, info, out_dir, data_path):
    """Save model + metadata to out_dir/<run_id>/ and point out_dir/latest.txt to it."""
    run_id = datetime.now().strftime("catboost_%Y%m%d_%H%M%S")
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
        **info,
        "data": {"file": data_path.name, "sha256": hashlib.sha256(data_path.read_bytes()).hexdigest()},
        "versions": {"python": platform.python_version(), "catboost": catboost.__version__,
                     "scikit-learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (Path(out_dir) / "latest.txt").write_text(run_id + "\n", encoding="utf-8")
    return run_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the diabetes screening model.")
    parser.add_argument("--data", default=DEFAULT_DATA, help="CSV in the BRFSS 2015 Diabetes_012 format")
    parser.add_argument("--out", default=DEFAULT_OUT, help="folder for trained models (default: models/)")
    args = parser.parse_args(argv)

    print(f"Reading {args.data}")
    model, info = train(pd.read_csv(args.data))
    run_dir = save(model, info, args.out, args.data)

    test = info["metrics"]["test"]
    print(f"\nTest set: ROC-AUC {test['roc_auc']:.4f}, recall {test['recall']:.1%}, "
          f"precision {test['precision']:.1%}, flagged {test['share_flagged']:.1%}")
    print(f"Saved to {run_dir}")


if __name__ == "__main__":
    sys.exit(main())
