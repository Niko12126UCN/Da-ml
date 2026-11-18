# Da-ml — Diabetes Risk Screening

A machine learning pipeline that estimates a person's risk of **prediabetes or diabetes** from 21 survey answers
and tells them whether it's worth talking to a doctor. Trained on the CDC BRFSS 2015 survey
([Kaggle — Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset/data)).

> **This is a screening tool, not a diagnosis.** It estimates risk for people with similar answers; it can't measure
> anything about a person's body.

| Document | Contents |
|---|---|
| [`PROJECT_PLAN.md`](PROJECT_PLAN.md) | Goal, steps and status |
| [`DECISIONS.md`](DECISIONS.md) | Every decision: what, why, when |
| [`RESULTS.md`](RESULTS.md) | All measured results, step by step |
| [`MODEL_EXPLAINED.md`](MODEL_EXPLAINED.md) | How a prediction is made, worked example, limits |
| [`Diabetes_Prediction_Feasibility.md`](Diabetes_Prediction_Feasibility.md) | Expectations before training; CatBoost explained |

## Setup

Requires [Miniconda](https://docs.conda.io/en/latest/miniconda.html). From the project root:

```
conda env create -f environment.yml
conda activate da-ml
```

In PyCharm, select the interpreter with **Add Interpreter → Select existing** →
`C:\Users\<you>\miniconda3\envs\da-ml\python.exe` (not "Generate new", which overwrites the environment).

## Run the app

**Easiest:** double-click **`start.bat`** (or run `python run.py` in the `da-ml` environment). It starts the API and
the web UI, opens http://127.0.0.1:8050 in your browser, and stops both when you press Ctrl+C in its window.
`python run.py --check` only checks that everything starts, then stops again.

Or start the two servers yourself, each in its own terminal (both in the `da-ml` environment, from the project root):

```
python -m api.main     # prediction API  -> http://127.0.0.1:5001
python -m ui.app       # web UI          -> http://127.0.0.1:8050
```

Then open **http://127.0.0.1:8050** in a browser. The UI needs the API to be running. The repository contains a
trained model (`models/latest.txt` names it), so no training is needed first.

## Train a new model

```
python -m ml.train                            # the Kaggle CSV in data/raw_data/
python -m ml.train --data path/to/file.csv    # any CSV in the same Diabetes_012 format
python -m ml.train --data ... --force         # deploy even if the quality check fails
```

In PyCharm: choose **Train model** in the run menu (top right) and press ▶. To use another file, change the
path under **Run → Edit Configurations… → Train model → Parameters**.

This validates the data, trains CatBoost, picks the threshold for 80% recall, evaluates on a 20% test split and saves
the model to a new folder `models/catboost_<date>_<time>/`. Then a **quality check** decides whether it goes live:
test ROC-AUC at least 0.80 and not more than 0.01 below the current model, and recall at least 75%.

- **Passed:** `models/latest.txt` points to the new model, and a **running app switches to it by itself**; no
  restart needed. Exit code 0.
- **Failed:** the model is saved (marked "not deployed"), the current model stays live, and the reasons are
  printed. Exit code 2.

To go back to an earlier model, write its folder name into `models/latest.txt`; the running app switches to it too.
The page header in the UI always shows which model is live (it updates within 10 seconds of a switch).

## Data store

`data/store.db` (SQLite) records every training run, and the answers + result of every person who ticks **"Help
improve the model: store my answers … anonymously"** in the UI (opt-in; nothing is stored otherwise). No name, IP
address or browser ID is stored. The file contains health data and is **never committed** (it is in `.gitignore`).
Other API clients can ask for storage with `POST /predict?store=true`. To use another file, set the environment
variable `DA_ML_STORE`.

## Monitoring

```
python -m ml.monitoring
```

Shows the live model's request volume and error rate, and compares the stored answers (opt-in) with the data the
model was trained on: per question and for the predictions, PSI < 0.1 is stable, 0.1–0.25 "watch", ≥ 0.25 "drift".
At least 100 stored predictions of the live model are needed. Each model has a `reference.json` for this; for a model
without one: `python -m ml.monitoring --build-reference <run_id> --data <csv it was trained on>`.

## Tests

```
python -m pytest tests
python -m flake8 --max-line-length 120 ml api ui tests
```

## Project structure

```
data/raw_data/        Kaggle CSVs              ml/validation.py   the dataset format + input checks
data/splits/          shared train/test split  ml/train.py        training script
ml/notebooks/         EDA, Modeling, AutoML    ml/risk.py         risk levels and messages
models/               trained models           api/main.py        Flask API
tests/                pytest tests             ui/app.py          Dash web UI
```

The AutoML notebook (`ml/notebooks/AutoML.ipynb`) uses a separate environment: `conda env create -f
environment-automl.yml`.
