# Project Plan — Diabetes Risk Screening Pipeline

**Dataset:** [Kaggle — Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset/data)
(BRFSS 2015, `diabetes_012_health_indicators_BRFSS2015.csv`)
**Background:** see [`Diabetes_Prediction_Feasibility.md`](Diabetes_Prediction_Feasibility.md) for why prediction is
feasible, expected performance, and an explanation of CatBoost.
**Decisions:** [`DECISIONS.md`](DECISIONS.md) · **Measured results:** [`RESULTS.md`](RESULTS.md) ·
**How the model works:** [`MODEL_EXPLAINED.md`](MODEL_EXPLAINED.md)

## 1. Goal

Build a complete pipeline that can:

1. **Take a dataset in this exact format** (the BRFSS 2015 columns), validate it, and reject bad input.
2. **Train a model** that predicts the risk of diabetes or prediabetes.
3. **Deploy the model** behind an API.
4. **Host a web UI** where a person enters their answers and gets a risk prediction.

> **This is a screening tool, not a diagnosis.** Its purpose is to find **people who should see a doctor about their
> diabetes risk**; the doctor decides whether and how to test (D-49). Every design decision below follows from that.

## 2. Status

| Step | Status |
|---|---|
| Data downloaded to `data/raw_data/` | ✅ Done |
| Environment (`environment.yml`, conda env `da-ml`) | ✅ Done |
| EDA + YData Profiling report (`ml/notebooks/EDA.ipynb`, `Diabetes_EDA_Profile.html`) | ✅ Done |
| Feasibility assessment (`Diabetes_Prediction_Feasibility.md`) | ✅ Done |
| Model comparison (`ml/notebooks/Modeling.ipynb`): CatBoost chosen, 80% recall | ✅ Done (2026-11-17) |
| AutoML benchmark (`ml/notebooks/AutoML.ipynb`): AutoGluon +0.001 ROC-AUC, CatBoost kept | ✅ Done (2026-11-17) |
| Shared input validation (`ml/validation.py`, `tests/test_validation.py`) | ✅ Done (2026-11-18) |
| Training script (`ml/train.py`, `tests/test_train.py`) → `models/catboost_20261118_084406/` | ✅ Done (2026-11-18) |
| Risk levels + API (`ml/risk.py`, `api/main.py`, `tests/test_api.py`) | ✅ Done (2026-11-18) |
| Web UI (`ui/app.py`, `ui/assets/style.css`, `tests/test_ui.py`) + README | ✅ Done (2026-11-18) |
| Local deployment (`run.py`, `start.bat`): runs on this PC | ✅ Done (2026-11-18) |
| Automatic deployment: quality check + live switch without restart (`ml/train.py`, `api/main.py`) | ✅ Done (2026-11-18) |
| Hosting for others (Docker / cloud) | ⬜ Later (D-60) |
| **Phase 2 (D-66)** | |
| 9. Data store + logging (`ml/store.py`, `data/store.db`; opt-in in the UI) | ✅ Done (2026-11-18) |
| 10. Monitoring: data drift, prediction drift, volume, errors per model (`ml/monitoring.py`) | ⬜ Next |
| 11. Real-time dashboard (`/dashboard`) | ⬜ |
| 12. Closing the loop: drift alert → "retraining recommended"; model history | ⬜ |

## 3. Model selection approach

The model choice only needs to be **argued for and tested**, not exhaustive. The plan:

| Step | Method | Purpose |
|---|---|---|
| 0 | **Dummy classifier** (always "no diabetes") | The floor any real model must beat (~84% accuracy) |
| 1 | **Logistic regression** (manual) | Simple, fully explainable linear baseline. Contrasts best with CatBoost (linear vs. trees) |
| 2 | **CatBoost** (manual) | Main candidate: gradient-boosted trees with native handling of categorical features |
| 3 | **AutoML** (benchmark) | Tries many models automatically to check whether something clearly better exists |

### AutoML

- **Candidate tools:** **AutoGluon** (usually the strongest; includes CatBoost, LightGBM, XGBoost, random forest,
  neural networks and stacked ensembles) or **PyCaret** (clear `compare_models()` leaderboard; good for presenting).
  auto-sklearn does not run on Windows; H2O does not include CatBoost and needs Java.
- **Run in a separate conda environment.** AutoML packages may conflict with the pinned versions in
  `environment.yml` (scikit-learn 1.4, numpy 1.26).
- **Use the exact same train/test split and test set** as the manual models.
- **For deployment**, keep only the single best model (AutoGluon: `refit_full` /
  `presets="optimize_for_deployment"`). A full stacked ensemble is large and slow to serve.

### Decision rule (agreed **before** seeing results)

> We switch to the AutoML model **only if it beats CatBoost by more than 0.01 ROC-AUC**, or clearly improves recall at
> our chosen threshold, on the same test set. Otherwise we deploy CatBoost (or logistic regression, if it is within
> ~0.01 of CatBoost, since it is simpler to explain).

**Actual outcome (2026-11-17):** as expected. CatBoost beat logistic regression by 0.006 ROC-AUC (0.825 vs. 0.819),
and AutoGluon's best model beat CatBoost by only 0.0012, so **CatBoost is kept** (D-22, D-36). Full numbers in
[`RESULTS.md`](RESULTS.md).

## 4. Data preparation

- **Target:** convert `Diabetes_012` to a yes/no target: `0` = no diabetes, `1` = prediabetes **or** diabetes.
  Prediabetes alone is only 1.8% of the data and too rare to predict as its own class. The pipeline does this
  conversion itself, so the raw Kaggle file can be used as-is.
- **Split first:** stratified train/test split (e.g. 80/20, fixed random seed). The test set is touched only once,
  at the very end.
- **Balancing only the training data:**
  - Option A: **class weights** (`class_weight="balanced"` / `auto_class_weights="Balanced"`), which keeps all data.
  - Option B: **50/50 downsampling** of the training data (~70k rows is still plenty).
  - Try both and compare. → **Result: neither is used** (D-19), see `DECISIONS.md`.
  - **Never balance the test set.** It must keep the real ratio (~84% / 16%), or the scores, especially precision,
    look better than they would be in practice.
- **Duplicates:** decide how to handle the ~24k duplicate rows **before** splitting, so identical rows can't end up
  in both train and test.
- **Preprocessing:** logistic regression needs scaled numeric features (`StandardScaler`) and ideally one-hot encoded
  categories; CatBoost does not.
- **Cross-validation** (stratified, e.g. 5-fold) on the training data for comparing and tuning models
  (`optuna` is already in `environment.yml`).

## 5. Evaluation (screening focus)

| Metric | Role |
|---|---|
| **Recall** at the chosen threshold | **Main result:** what share of people with diabetes are caught |
| **Precision** at the same threshold | What share of flagged people actually have diabetes |
| **ROC-AUC** | Main metric for comparing models (independent of threshold) |
| **PR-AUC** | Performance on the minority class |
| **F1** | Balance of recall and precision |
| **Accuracy** | Reported only next to the dummy baseline (~84%) |
| **Training / prediction time, model size** | Matters for deployment |

- **Threshold chosen for recall**, e.g. catch at least ~80% of diabetics and accept more false alarms. A false alarm
  costs one blood test; a missed case can go undiagnosed for years. Expected: ~75–80% recall at ~30–35% precision.
- **Calibrated probabilities:** if the model is trained on 50/50 data it will overestimate risk, so calibrate it
  (or use class weights) so that "62% risk" in the UI means a real 62%.
- The chosen **threshold is saved together with the model**, so the API uses exactly the same one.

## 6. Input validation (shared by training and the API)

One module (`ml/validation.py`) used in **both** `train.py` and the API, based on the checks in `EDA.ipynb` section 6:

- exactly the expected 22 columns (21 features + `Diabetes_012` for training; 21 features for prediction),
- binary columns contain only 0/1,
- coded columns are in range: `GenHlth` 1–5, `Age` 1–13, `Education` 1–6, `Income` 1–8, `MentHlth`/`PhysHlth` 0–30,
  `Diabetes_012` 0–2,
- `BMI` within a plausible range,
- no missing values.

Bad input is **rejected with a clear error message**, never silently used.

## 7. Web UI

- **Built (D-52 to D-57):** question cards with tap-to-select buttons, BMI from height and weight, all questions
  required, result panel with a risk bar. Run with `python -m ui.app` (needs the API, see `README.md`).
- **Human-readable inputs:** dropdowns and toggles with labels ("60–64", "$25,000–35,000", "Excellent … Poor")
  that map to the codes behind the scenes. The mappings are in the feature table in `EDA.ipynb`.
- All 21 inputs are questions a person can answer about themselves, so the form works as a self-check.
- **Output:** a risk level (**low / elevated / high**, D-48) plus the calibrated probability. Never a yes/no
  "you have diabetes". The API returns the level, the message and the disclaimer (D-50).
- **Messages recommend seeing a doctor**, never a specific test (D-49), e.g. for high risk:
  *"Your estimated risk is high (33%), about twice the average or more. We recommend that you see your doctor about
  your diabetes risk."* Every result also says: *"This is a screening estimate based on survey answers, not a
  diagnosis."*

## 8. Project structure

Mirrors the `ml-taxi-pipeline` reference project (Flask API + Dash UI):

```
Da-ml/
├── data/raw_data/                 # raw Kaggle CSVs                                  ✅
├── ml/
│   ├── notebooks/EDA.ipynb        # EDA + profiling report                           ✅
│   ├── notebooks/Modeling.ipynb   # LR vs CatBoost vs AutoML comparison + reasoning
│   ├── train.py                   # validate → train → save model + metrics + threshold    ✅
│   ├── validation.py              # shared schema checks (train.py AND the API)          ✅
│   └── risk.py                    # risk levels (low/elevated/high) + messages            ✅
├── tests/                         # pytest: test_validation.py, test_train.py            ✅
├── ml/store.py / data/store.db    # data store: predictions (opt-in) + training runs (not in git) ✅
├── models/<run_id>/               # model.cbm + metadata.json per run; latest.txt = current ✅
├── api/main.py                    # Flask: /health, /model, /predict (port 5001)           ✅
├── ui/app.py                      # Dash page → calls /predict → shows risk level (port 8050) ✅
├── run.py / start.bat             # start API + UI together on this PC                ✅
├── environment.yml                # da-ml environment                                ✅
├── Diabetes_Prediction_Feasibility.md                                                ✅
├── DECISIONS.md                   # every decision: what, why, when                  ✅
├── RESULTS.md                     # all measured results, step by step               ✅
├── MODEL_EXPLAINED.md             # how a prediction is made, worked example, limits  ✅
└── PROJECT_PLAN.md                # this file
```

- **The notebook argues, the script runs.** `Modeling.ipynb` documents *why* a model was chosen; `train.py` is the
  repeatable pipeline step (fixed seed, saved metrics, saved threshold).

## Phase 2 — towards the full production pipeline (D-66)

Target picture: a production pipeline where new data and predictions go into a **data store**, a **training pipeline**
produces **stored models**, an **inference pipeline** serves them, and **monitoring** and a **real-time dashboard**
watch data and model and feed back into retraining.

| Drawing | Here | Status |
|---|---|---|
| IoT data (new data) | People using the web UI; CSV files | ✅ |
| Data store | `data/store.db` (SQLite): stored predictions (opt-in) + training runs | ✅ step 9 |
| Historical data → training pipeline | `python -m ml.train` (labelled CSV files) | ✅ |
| Stored models | `models/<run_id>/`, `latest.txt`, quality check | ✅ |
| Inference pipeline | API with automatic model switch | ✅ |
| Predictions → data store | `/predict?store=true` (UI: opt-in checkbox) | ✅ step 9 |
| Data & model monitoring | drift of answers vs. training data, prediction distribution, share flagged, volume, errors | ⬜ step 10 |
| Real-time dashboard | `/dashboard` page, refreshing by itself | ⬜ step 11 |
| Feedback into training | drift alert → "retraining recommended"; training runs in the store | ⬜ step 12 |

**Limitation:** the store gets **no true labels** (users don't report a diagnosis), so live accuracy/recall can't be
measured and stored predictions can't be used for retraining. Retraining needs labelled datasets.

**Privacy (D-67 to D-70):** only with opt-in consent; no identifying fields; kept indefinitely for research;
`data/store.db` never in git.

## 9. Open questions

- ~~**Deployment target:** local, Docker or hosted?~~ → **Local for now** (D-59); hosting decided later (D-60).
  Original question: local (API + UI on one machine), Docker container (Docker Desktop is installed), or hosted
  (e.g. Azure, Render)? This decides how much model size and dependencies matter, especially for an AutoML model.
- ~~**Duplicates:** keep or drop the ~24k duplicate rows?~~ → **Keep** (D-17)
- ~~**Balancing:** class weights or 50/50 downsampling?~~ → **Neither**; use the threshold (D-19)
- ~~**Exact recall target** for the threshold?~~ → **80%** (D-23)
- ~~**Which AutoML tool:** AutoGluon or PyCaret?~~ → **AutoGluon** (D-27); result: CatBoost kept (D-36)
