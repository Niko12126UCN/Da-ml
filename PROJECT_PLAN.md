# Project Plan — Diabetes Risk Screening Pipeline

**Dataset:** [Kaggle — Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset/data)
(BRFSS 2015, `diabetes_012_health_indicators_BRFSS2015.csv`)
**Background:** see [`Diabetes_Prediction_Feasibility.md`](Diabetes_Prediction_Feasibility.md) for why prediction is
feasible, expected performance, and an explanation of CatBoost.

## 1. Goal

Build a complete pipeline that can:

1. **Take a dataset in this exact format** (the BRFSS 2015 columns), validate it, and reject bad input.
2. **Train a model** that predicts the risk of diabetes or prediabetes.
3. **Deploy the model** behind an API.
4. **Host a web UI** where a person enters their answers and gets a risk prediction.

> **This is a screening tool, not a diagnosis.** Its purpose is to find **candidates for an actual diabetes test**
> (blood glucose / HbA1c). Every design decision below follows from that.

## 2. Status

| Step | Status |
|---|---|
| Data downloaded to `data/raw_data/` | ✅ Done |
| Environment (`environment.yml`, conda env `da-ml`) | ✅ Done |
| EDA + YData Profiling report (`ml/notebooks/EDA.ipynb`, `Diabetes_EDA_Profile.html`) | ✅ Done |
| Feasibility assessment (`Diabetes_Prediction_Feasibility.md`) | ✅ Done |
| Model comparison (`ml/notebooks/Modeling.ipynb`): CatBoost chosen, 80% recall | ✅ Done (2026-11-17) |
| AutoML benchmark | ⬜ Next |
| Training script + validation (`ml/train.py`, `ml/validation.py`) | ⬜ |
| API (`api/main.py`) | ⬜ |
| Web UI (`ui/app.py`) | ⬜ |
| Deployment | ⬜ (target not decided, see section 9) |

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

Expected outcome: models on this data usually score within ~0.01–0.02 ROC-AUC of each other (around 0.80–0.83),
so CatBoost will most likely be kept.

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

- **Human-readable inputs:** dropdowns and toggles with labels ("60–64", "$25,000–35,000", "Excellent … Poor")
  that map to the codes behind the scenes. The mappings are in the feature table in `EDA.ipynb`.
- All 21 inputs are questions a person can answer about themselves, so the form works as a self-check.
- **Output:** a risk level (**low / elevated / high**) plus the calibrated probability. Never a yes/no
  "you have diabetes".
- **Clear disclaimer** on the result, e.g.:
  *"Elevated risk — consider a blood glucose / HbA1c test with your doctor. This is not a diagnosis."*

## 8. Project structure

Mirrors the `ml-taxi-pipeline` reference project (Flask API + Dash UI):

```
Da-ml/
├── data/raw_data/                 # raw Kaggle CSVs                                  ✅
├── ml/
│   ├── notebooks/EDA.ipynb        # EDA + profiling report                           ✅
│   ├── notebooks/Modeling.ipynb   # LR vs CatBoost vs AutoML comparison + reasoning
│   ├── train.py                   # validate → train chosen model → save model + metrics + threshold
│   └── validation.py              # shared schema checks (train.py AND the API)
├── models/                        # saved model, metrics and threshold (versioned)
├── api/main.py                    # Flask: /health, /predict (loads model, validates input)
├── ui/app.py                      # Dash form → calls /predict → shows risk level + disclaimer
├── environment.yml                # da-ml environment                                ✅
├── Diabetes_Prediction_Feasibility.md                                                ✅
└── PROJECT_PLAN.md                # this file
```

- **The notebook argues, the script runs.** `Modeling.ipynb` documents *why* a model was chosen; `train.py` is the
  repeatable pipeline step (fixed seed, saved metrics, saved threshold).

## 9. Open questions

- **Deployment target:** local (API + UI on one machine), Docker container (Docker Desktop is installed), or hosted
  (e.g. Azure, Render)? This decides how much model size and dependencies matter, especially for an AutoML model.
- ~~**Duplicates:** keep or drop the ~24k duplicate rows?~~ → **Keep** (D-17)
- ~~**Balancing:** class weights or 50/50 downsampling?~~ → **Neither**; use the threshold (D-19)
- ~~**Exact recall target** for the threshold?~~ → **80%** (D-23)
- **Which AutoML tool:** AutoGluon (strongest) or PyCaret (clearest overview)?
