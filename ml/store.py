"""Data store (step 9 of PROJECT_PLAN.md): a SQLite database with every stored prediction and every training run.

Tables:
    predictions    - one row per person whose answers were stored (opt-in only, D-68): time (UTC), model, probability,
                     risk level, flagged, source, and the 21 answers. No name, IP address or browser ID (D-67).
    training_runs  - one row per trained model (deployed or not), imported from models/<run_id>/metadata.json.

The database file is data/store.db (override with the environment variable DA_ML_STORE). It holds health data about
individuals and is not committed to git (D-70). Records are kept indefinitely for research (D-69).
"""

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ml.validation import FEATURE_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STORE = Path(os.environ.get("DA_ML_STORE", PROJECT_ROOT / "data" / "store.db"))

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS predictions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   TEXT    NOT NULL,              -- UTC, ISO 8601
    model        TEXT    NOT NULL,              -- run ID of the model that made the prediction
    probability  REAL    NOT NULL,
    risk_level   TEXT    NOT NULL,              -- low / elevated / high
    flagged      INTEGER NOT NULL,              -- 1 = elevated or high
    source       TEXT    NOT NULL,              -- 'ui' or 'api'
    {", ".join(f"{c} INTEGER NOT NULL" for c in FEATURE_COLUMNS)}
);
CREATE INDEX IF NOT EXISTS predictions_created_at ON predictions (created_at);
CREATE INDEX IF NOT EXISTS predictions_model ON predictions (model);

CREATE TABLE IF NOT EXISTS training_runs (
    run_id          TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,              -- local time, as in metadata.json
    data_file       TEXT,
    data_sha256     TEXT,
    rows_total      INTEGER,
    threshold       REAL,
    test_roc_auc    REAL,
    test_recall     REAL,
    test_precision  REAL,
    deployed        INTEGER,                    -- 1 = passed the quality check (or forced) and went live
    forced          INTEGER,
    failures        TEXT                        -- JSON list of failed quality checks
);
"""


def connect(path=DEFAULT_STORE):
    """Open the store (creating the file and tables if needed)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")   # the API can write while the dashboard reads
    conn.executescript(SCHEMA)
    return conn


def log_predictions(X, results, model, source, path=DEFAULT_STORE):
    """Store predictions. X: validated features (DataFrame, one row per person); results: the assess() dicts."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    columns = ["created_at", "model", "probability", "risk_level", "flagged", "source", *FEATURE_COLUMNS]
    rows = [(now, model, r["probability"], r["risk_level"], int(r["flagged"]), source,
             *(int(v) for v in features))
            for features, r in zip(X[FEATURE_COLUMNS].itertuples(index=False), results)]
    with connect(path) as conn:
        conn.executemany(f"INSERT INTO predictions ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))})",
                         rows)
    conn.close()
    return len(rows)


def import_training_runs(models_dir, path=DEFAULT_STORE):
    """Add every model folder in models_dir that isn't in the store yet. Returns the number of runs added."""
    added = 0
    with connect(path) as conn:
        known = {row[0] for row in conn.execute("SELECT run_id FROM training_runs")}
        for metadata_file in sorted(Path(models_dir).glob("*/metadata.json")):
            m = json.loads(metadata_file.read_text(encoding="utf-8"))
            if "run_id" not in m or m["run_id"] in known:
                continue
            test = m.get("metrics", {}).get("test", {})
            quality = m.get("quality_check") or {}
            conn.execute(
                "INSERT INTO training_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (m["run_id"], m.get("created"), m.get("data", {}).get("file"), m.get("data", {}).get("sha256"),
                 m.get("rows", {}).get("total"), m.get("threshold"), test.get("roc_auc"), test.get("recall"),
                 test.get("precision"),
                 # models from before the quality check (D-61) were all deployed
                 int(quality.get("deployed", True)), int(quality.get("forced", False)),
                 json.dumps(quality.get("failures", []))))
            added += 1
    conn.close()
    return added


def read_table(table, path=DEFAULT_STORE):
    """A whole table as a DataFrame (for monitoring and the dashboard)."""
    if table not in {"predictions", "training_runs"}:
        raise ValueError(table)
    with connect(path) as conn:
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    conn.close()
    return df
