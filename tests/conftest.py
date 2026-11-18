"""Test setup: tests must never write to the real data store (data/store.db, D-70).

DA_ML_STORE is set before any project module is imported, so the default store path of ml.store, the API and the
training script points to a temporary file for the whole test run.
"""

import os
import tempfile
from pathlib import Path

os.environ["DA_ML_STORE"] = str(Path(tempfile.mkdtemp(prefix="da-ml-tests-")) / "store.db")
