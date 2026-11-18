"""Data store (step 9 of PROJECT_PLAN.md): a SQLite database with every stored prediction and every training run.

Tables:
    predictions    - one row per person whose answers were stored (opt-in only, D-68): time (UTC), model, probability,
                     risk level, flagged, source, the 21 answers, and the optional diagnosis the person gave (the
                     label for retraining, D-79). No name, IP address or browser ID (D-67).
    requests       - one row per /predict call: time, model, number of people, HTTP status, stored or not. No answers
                     (D-75); used to monitor volume and errors.
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
    {", ".join(f"{c} INTEGER NOT NULL" for c in FEATURE_COLUMNS)},
    diagnosis    INTEGER                        -- optional label, like Diabetes_012: 0 no, 1 prediabetes, 2 diabetes
                                                -- (NULL = not answered); never used for the prediction (D-79)
);
CREATE INDEX IF NOT EXISTS predictions_created_at ON predictions (created_at);
CREATE INDEX IF NOT EXISTS predictions_model ON predictions (model);

CREATE TABLE IF NOT EXISTS requests (          -- every /predict call; counts only, no answers (D-75)
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   TEXT    NOT NULL,              -- UTC, ISO 8601
    model        TEXT    NOT NULL,
    n_people     INTEGER NOT NULL,              -- people in the request (0 if it couldn't be read)
    status       INTEGER NOT NULL,              -- HTTP status: 200 = OK, 400 = invalid input
    stored       INTEGER NOT NULL               -- 1 = the answers were stored (opt-in)
);
CREATE INDEX IF NOT EXISTS requests_created_at ON requests (created_at);

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
    failures        TEXT,                       -- JSON list of failed quality checks
    user_rows       INTEGER                     -- labelled user rows in the training data (D-80)
);
"""


def connect(path=DEFAULT_STORE):
    """Open the store (creating the file and tables if needed)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")   # the API can write while the dashboard reads
    conn.executescript(SCHEMA)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(predictions)")}
    if "diagnosis" not in columns:   # stores created before D-79
        conn.execute("ALTER TABLE predictions ADD COLUMN diagnosis INTEGER")
    if "user_rows" not in {row[1] for row in conn.execute("PRAGMA table_info(training_runs)")}:   # before D-80
        conn.execute("ALTER TABLE training_runs ADD COLUMN user_rows INTEGER")
    return conn


def log_predictions(X, results, model, source, path=DEFAULT_STORE, diagnosis=None):
    """Store predictions. X: validated features (DataFrame, one row per person); results: the assess() dicts;
    diagnosis: the optional label (0/1/2) the person gave, or None (D-79)."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    columns = ["created_at", "model", "probability", "risk_level", "flagged", "source", *FEATURE_COLUMNS, "diagnosis"]
    rows = [(now, model, r["probability"], r["risk_level"], int(r["flagged"]), source,
             *(int(v) for v in features), diagnosis)
            for features, r in zip(X[FEATURE_COLUMNS].itertuples(index=False), results)]
    with connect(path) as conn:
        conn.executemany(f"INSERT INTO predictions ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))})",
                         rows)
    conn.close()
    return len(rows)


def log_request(model, n_people, status, stored, path=DEFAULT_STORE):
    """Count one /predict call (for volume and error monitoring). No answers are stored here."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with connect(path) as conn:
        conn.execute("INSERT INTO requests (created_at, model, n_people, status, stored) VALUES (?, ?, ?, ?, ?)",
                     (now, model, int(n_people), int(status), int(stored)))
    conn.close()


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
                "INSERT INTO training_runs (run_id, created_at, data_file, data_sha256, rows_total, threshold, "
                "test_roc_auc, test_recall, test_precision, deployed, forced, failures, user_rows) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (m["run_id"], m.get("created"), m.get("data", {}).get("file"), m.get("data", {}).get("sha256"),
                 m.get("rows", {}).get("total"), m.get("threshold"), test.get("roc_auc"), test.get("recall"),
                 test.get("precision"),
                 # models from before the quality check (D-61) were all deployed
                 int(quality.get("deployed", True)), int(quality.get("forced", False)),
                 json.dumps(quality.get("failures", [])), m.get("rows", {}).get("user_rows", 0)))
            added += 1
    conn.close()
    return added


def labelled_user_rows(path=DEFAULT_STORE):
    """Stored predictions with a diagnosis, in the training format (21 features + Diabetes_012), for retraining
    together with the original dataset (D-80)."""
    df = read_table("predictions", path)
    df = df[df["diagnosis"].notna()]
    return df[FEATURE_COLUMNS].assign(Diabetes_012=df["diagnosis"].astype(int)).reset_index(drop=True)


def read_table(table, path=DEFAULT_STORE):
    """A whole table as a DataFrame (for monitoring and the dashboard)."""
    if table not in {"predictions", "requests", "training_runs"}:
        raise ValueError(table)
    with connect(path) as conn:
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    conn.close()
    return df
