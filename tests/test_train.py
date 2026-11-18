import json
from pathlib import Path

import pandas as pd
import pytest
from catboost import CatBoostClassifier

import ml.train
from ml.train import DEFAULT_DATA, activate, main, make_target, quality_check, save, split, train
from ml.validation import FEATURE_COLUMNS, ValidationError, validate_training_data

SPLIT_FILE = Path(__file__).resolve().parents[1] / "data" / "splits" / "train_test_split.csv"
FAST_PARAMS = {"iterations": 20, "learning_rate": 0.3, "depth": 4}


@pytest.fixture(scope="module")
def raw():
    return pd.read_csv(DEFAULT_DATA)


def test_split_matches_shared_split_file(raw):
    df = validate_training_data(raw)
    _, X_test, _, _ = split(df[FEATURE_COLUMNS], make_target(df))
    shared = pd.read_csv(SPLIT_FILE, index_col="row")["split"]
    assert set(X_test.index) == set(shared[shared == "test"].index)


def test_train_and_save_end_to_end(raw, tmp_path):
    sample = raw.sample(5000, random_state=0)
    sample_csv = tmp_path / "sample.csv"
    sample.to_csv(sample_csv, index=False)

    model, info = train(sample, params=FAST_PARAMS, n_folds=3, log=lambda *_: None)
    assert 0 < info["threshold"] < 1
    assert info["metrics"]["cross_validation"]["recall"] >= 0.80     # threshold chosen for 80% recall
    assert info["metrics"]["test"]["roc_auc"] > 0.7
    assert info["rows"] == {"total": 5000, "train": 4000, "test": 1000, "positive_share": pytest.approx(
        (sample["Diabetes_012"] > 0).mean())}

    run_dir = save(model, info, tmp_path / "models", sample_csv)
    assert not (tmp_path / "models" / "latest.txt").exists()          # saving alone doesn't make it live
    activate(tmp_path / "models", run_dir.name)
    assert (tmp_path / "models" / "latest.txt").read_text().strip() == run_dir.name
    assert not (tmp_path / "models" / "latest.tmp").exists()
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["threshold"] == info["threshold"]
    assert metadata["features"] == FEATURE_COLUMNS
    assert len(metadata["data"]["sha256"]) == 64

    loaded = CatBoostClassifier().load_model(str(run_dir / "model.cbm"))
    X = validate_training_data(sample)[FEATURE_COLUMNS]
    assert (loaded.predict_proba(X) == model.predict_proba(X)).all()


def fake_info(roc_auc, recall):
    return {"metrics": {"test": {"roc_auc": roc_auc, "recall": recall}}}


def live(roc_auc, run_id="catboost_live"):
    return {"run_id": run_id, **fake_info(roc_auc, 0.8)}


@pytest.mark.parametrize("roc_auc, recall, current, n_failures", [
    (0.825, 0.80, None, 0),          # first model, good
    (0.825, 0.80, live(0.830), 0),   # slightly worse than live (within 0.01): fine
    (0.815, 0.80, live(0.830), 1),   # more than 0.01 worse than live
    (0.79, 0.80, None, 1),           # below the absolute minimum
    (0.825, 0.70, None, 1),          # recall too low
    (0.70, 0.60, live(0.83), 3),     # everything wrong
])
def test_quality_check(roc_auc, recall, current, n_failures):
    assert len(quality_check(fake_info(roc_auc, recall), current)) == n_failures


@pytest.fixture
def quick_training(raw, tmp_path, monkeypatch):
    """Run main() fast on a small sample, with the test ROC-AUC set to a chosen value."""
    sample_csv = tmp_path / "sample.csv"
    raw.sample(3000, random_state=0).to_csv(sample_csv, index=False)
    real_train = ml.train.train

    def run(roc_auc, *flags):
        def fast_train(df):
            model, info = real_train(df, params=FAST_PARAMS, n_folds=3, log=lambda *_: None)
            info["metrics"]["test"].update(roc_auc=roc_auc, recall=0.8)
            return model, info
        monkeypatch.setattr(ml.train, "train", fast_train)
        return main(["--data", str(sample_csv), "--out", str(tmp_path / "models"),
                     "--store", str(tmp_path / "store.db"), *flags])

    return run


def latest(tmp_path):
    return (tmp_path / "models" / "latest.txt").read_text().strip()


def run_metadata(tmp_path, run_id):
    return json.loads((tmp_path / "models" / run_id / "metadata.json").read_text(encoding="utf-8"))


def run_ids(tmp_path):
    return {p.name for p in (tmp_path / "models").iterdir() if p.is_dir()}


def test_good_model_goes_live_and_worse_model_does_not(quick_training, tmp_path):
    assert quick_training(0.825) == 0                       # first model: live
    first = latest(tmp_path)
    assert run_metadata(tmp_path, first)["quality_check"]["deployed"] is True

    assert quick_training(0.80) == 2                        # 0.025 worse: trained, saved, but NOT live
    assert latest(tmp_path) == first
    (rejected,) = run_ids(tmp_path) - {first}
    check = run_metadata(tmp_path, rejected)["quality_check"]
    assert check["deployed"] is False and check["compared_with"] == first and check["failures"]

    assert quick_training(0.80, "--force") == 0             # forced: live
    assert latest(tmp_path) not in {first, rejected}
    assert run_metadata(tmp_path, latest(tmp_path))["quality_check"]["forced"] is True

    from ml.store import read_table                         # all three runs are in the data store (D-72)
    runs = read_table("training_runs", tmp_path / "store.db").set_index("run_id")
    assert len(runs) == 3 and runs.loc[rejected, "deployed"] == 0 and runs.loc[first, "deployed"] == 1


def test_runs_in_the_same_second_get_different_folders(raw, tmp_path):
    sample_csv = tmp_path / "sample.csv"
    raw.head(10).to_csv(sample_csv, index=False)
    model, info = CatBoostClassifier(), {"metrics": {}}
    first = save(model.fit([[0], [1]], [0, 1], verbose=0), info, tmp_path / "models", sample_csv)
    second = save(model, info, tmp_path / "models", sample_csv)
    assert first != second and first.exists() and second.exists()


def test_invalid_data_is_rejected_before_training(raw):
    bad = raw.head(500).copy()
    bad.loc[0, "Age"] = 99
    with pytest.raises(ValidationError, match="'Age'"):
        train(bad, params=FAST_PARAMS, log=lambda *_: None)
