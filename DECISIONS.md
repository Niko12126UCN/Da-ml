# Decision Log

Every decision made in the project is recorded here: **what** was decided, **why**, and **when**.
Newest decisions are at the bottom. If a decision is changed later, the old entry stays and is marked
**Superseded by D-xx**, so the history is kept.

**Status values:** ✅ Decided · 🔶 Preliminary (may change after a later step) · ⛔ Superseded

| ID | Date | Decision | Status |
|---|---|---|---|
| D-01 | 2026-11-16 | Use the Kaggle BRFSS 2015 diabetes dataset | ✅ |
| D-02 | 2026-11-16 | Use the full three-class file for EDA and modeling | ✅ |
| D-03 | 2026-11-16 | Convert all columns from float to integer when loading | ✅ |
| D-04 | 2026-11-16 | Separate `da-ml` conda environment with the same pinned packages as the taxi project | ✅ |
| D-05 | 2026-11-16 | Pin `setuptools=80.9.0` and `ipython==9.15.0` | ✅ |
| D-06 | 2026-11-17 | Use `%matplotlib inline` in notebooks that use ydata-profiling | ✅ |
| D-07 | 2026-11-17 | The model is a screening tool, not a diagnosis | ✅ |
| D-08 | 2026-11-17 | Yes/no target: prediabetes merged with diabetes | ✅ |
| D-09 | 2026-11-17 | Compare logistic regression and CatBoost by hand, then AutoML as a benchmark | ✅ |
| D-10 | 2026-11-17 | Decision rule for switching to the AutoML model | ✅ |
| D-11 | 2026-11-17 | Run AutoML in a separate conda environment | ✅ |
| D-12 | 2026-11-17 | Balance only training data; never the test set | ✅ |
| D-13 | 2026-11-17 | Evaluate by recall/precision at a threshold and ROC-AUC, not accuracy | ✅ |
| D-14 | 2026-11-17 | Shared input validation for training and API | ✅ |
| D-15 | 2026-11-17 | Project structure mirrors the taxi project (Flask API + Dash UI) | ✅ |
| D-16 | 2026-11-17 | Build the pipeline one step at a time | ✅ |
| D-17 | 2026-11-17 | Keep duplicates (confirmed by test in `Modeling.ipynb`) | ✅ |
| D-18 | 2026-11-17 | Test both class weights and 50/50 downsampling | ⛔ Superseded by D-19 |
| D-19 | 2026-11-17 | No balancing; reach the recall target with the threshold instead | ✅ |
| D-20 | 2026-11-17 | CatBoost treats ordinal codes as numbers, not categories | ✅ |
| D-21 | 2026-11-17 | Compare models with default settings (no tuning) in step 1 | ✅ |
| D-22 | 2026-11-17 | CatBoost is the manual winner (the model AutoML must beat) | ✅ |
| D-23 | 2026-11-17 | Recall target 80% for the screening threshold | ✅ |

---

### D-01 — Use the Kaggle BRFSS 2015 diabetes dataset
**Date:** 2026-11-16
**Decision:** Use the [Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset/data)
instead of the generated taxi dataset used in the reference project.
**Why:** The taxi data was fake and generated for showcase only. This dataset is real (CDC BRFSS 2015 survey),
large (253,680 respondents) and already cleaned.

### D-02 — Use the full three-class file
**Date:** 2026-11-16
**Decision:** Use `diabetes_012_health_indicators_BRFSS2015.csv` (target `Diabetes_012`: 0/1/2). Don't use the
Kaggle 50/50 file for training.
**Why:** It is the most complete version (all 253,680 rows, all three classes). Any balancing is done by us on the
training data only (see D-12). The Kaggle 50/50 file is drawn from the same people, so its rows would overlap with
our test set.

### D-03 — Convert columns to integers
**Date:** 2026-11-16
**Decision:** Load the CSV with `.astype(int)`.
**Why:** All columns are stored as floats but contain only whole numbers (codes and counts). Integers make the
tables and the profiling report easier to read. No information is lost.

### D-04 — Separate `da-ml` conda environment
**Date:** 2026-11-16
**Decision:** Create a conda environment `da-ml` from `environment.yml`, with Python 3.11 and the same pinned
packages as the taxi project.
**Why:** Same setup as the reference project, and reproducible. The miniconda base environment (Python 3.14) had
none of the packages. Note: in PyCharm, select it with **"Select existing"**; "Generate new" overwrote it once.

### D-05 — Pin `setuptools` and `ipython`
**Date:** 2026-11-16
**Decision:** Add `setuptools=80.9.0` and `ipython==9.15.0` to `environment.yml`.
**Why:** ydata-profiling 4.16.1 breaks with newer versions. Newer setuptools removed `pkg_resources`, and IPython
9.18 raised `ImportError: cannot import name 'backend2gui'`. The versions match the working taxi environment.

### D-06 — `%matplotlib inline` in notebooks
**Date:** 2026-11-17
**Decision:** Set `%matplotlib inline` in the setup cell, and again after generating a profiling report.
**Why:** ydata-profiling switches matplotlib to the non-interactive `agg` backend and doesn't switch back, so
later plots weren't shown ("FigureCanvasAgg is non-interactive").

### D-07 — Screening tool, not a diagnosis
**Date:** 2026-11-17
**Decision:** The model estimates the **risk** of diabetes/prediabetes to find **candidates for an actual test**
(blood glucose / HbA1c). It never states that a person has diabetes.
**Why:** The data is a self-reported survey without lab values, and it is a snapshot (cross-sectional) rather than
a follow-up. It can rank people by risk but cannot diagnose. This shapes the metric (D-13), the threshold and the
UI wording.

### D-08 — Yes/no target
**Date:** 2026-11-17
**Decision:** Convert `Diabetes_012` to `0` = no diabetes, `1` = prediabetes **or** diabetes. The pipeline does the
conversion itself, so the raw Kaggle file can be used as-is.
**Why:** Prediabetes alone is only 1.8% of rows (4,631) and too rare to predict reliably. Many people with
prediabetes don't know it, so that label is noisy. For screening, both groups should be sent for a test anyway.

### D-09 — Logistic regression + CatBoost by hand, then AutoML
**Date:** 2026-11-17
**Decision:** Compare a dummy baseline, **logistic regression** and **CatBoost** by hand. Then run **AutoML** on the
same split as a benchmark to see if something clearly better exists.
**Why:** The project needs to show it can argue for and test methods, without testing everything by hand. Logistic
regression contrasts best with CatBoost (simple and explainable vs. powerful boosted trees). CatBoost suits the
mostly categorical/ordinal data. AutoML covers many more models with little effort.

### D-10 — Decision rule for AutoML
**Date:** 2026-11-17
**Decision:** Switch to the AutoML model **only if it beats CatBoost by more than 0.01 ROC-AUC**, or clearly
improves recall at the chosen threshold, on the same test set. If logistic regression is within ~0.01 of CatBoost,
it may be preferred because it is simpler to explain.
**Why:** The rule is agreed **before** seeing results, so the choice isn't just "whichever number was highest".
AutoML models (stacked ensembles) are larger, slower and harder to explain, so they must earn their place.

### D-11 — AutoML in a separate environment
**Date:** 2026-11-17
**Decision:** Install the AutoML tool in its own conda environment, not in `da-ml`.
**Why:** AutoML packages may require versions that conflict with the pins in `environment.yml` (scikit-learn 1.4,
numpy 1.26). Keeping them apart protects the working setup.

### D-12 — Balance only the training data
**Date:** 2026-11-17
**Decision:** Split first. Any balancing is applied only to the training data (in cross-validation: only to the
training part of each fold). Validation and test data always keep the real ratio (~84% / 16%).
**Why:** A balanced test set would show scores, especially precision, that are better than in practice. Real
users are ~16% positive, so the test must be too.

### D-13 — Evaluation metrics
**Date:** 2026-11-17
**Decision:** Main result: **recall and precision at a chosen threshold**. Model comparison: **ROC-AUC** (plus
PR-AUC). Accuracy is only reported next to the dummy baseline.
**Why:** A model that always says "no diabetes" is ~84% accurate but useless. For screening, what matters is how
many diabetics are caught (recall) and how many flagged people actually have diabetes (precision).

### D-14 — Shared input validation
**Date:** 2026-11-17
**Decision:** One validation module (`ml/validation.py`), used by both the training script and the API. It checks
columns, 0/1 flags, code ranges and missing values, and rejects bad input.
**Why:** The pipeline must accept "a dataset in this exact format". The same rules must apply to training data and
to a single prediction request, otherwise they can drift apart.

### D-15 — Project structure
**Date:** 2026-11-17
**Decision:** Mirror the `ml-taxi-pipeline` reference project: notebooks in `ml/notebooks/`, a training script,
saved models in `models/`, a Flask API in `api/` and a Dash UI in `ui/`.
**Why:** Same setup as the reference project. The notebook documents *why* a model was chosen; the script makes
training repeatable.

### D-16 — One step at a time
**Date:** 2026-11-17
**Decision:** Build the pipeline step by step in the order of `PROJECT_PLAN.md`, reviewing each step before the
next. Every decision is logged in this file, and open questions are asked rather than assumed.
**Why:** Each step builds on the previous one, and it's easier to check and discuss each part separately.

### D-17 — Keep duplicates
**Date:** 2026-11-17
**Decision:** Keep the ~24k duplicate rows. `Modeling.ipynb` tests whether they inflate the scores (grouped
cross-validation) and whether removing them from training helps.
**Why:** There is no ID column and most answers are yes/no, so different people easily give identical answers.
They are probably real respondents, not errors.
**Result (2026-11-17, `Modeling.ipynb` section 7):** confirmed. CatBoost ROC-AUC was 0.8244 with standard CV, 0.8248
with grouped CV (identical rows never split between train and validation) and 0.8238 with duplicates removed from
training. The differences are within noise, so duplicates neither inflate the scores nor hurt the model.

### D-18 — Test both balancing methods ⛔ Superseded by D-19
**Date:** 2026-11-17
**Decision:** Compare no balancing, class weights and 50/50 downsampling of the training data in `Modeling.ipynb`,
and choose based on the results.
**Why:** Both are common, and it's not obvious which is better here. Downsampling discards ~180k rows; class weights
keep all data. Both distort the predicted probabilities, which matters for the UI.

### D-19 — No balancing; use the threshold instead
**Date:** 2026-11-17
**Decision:** Train on the data as it is (no class weights, no downsampling). The recall target is reached by
lowering the decision threshold.
**Why:** Result of `Modeling.ipynb` section 6 (5-fold CV on the training data). Balancing did **not** improve how
well the models separate the classes. ROC-AUC for CatBoost was 0.8244 (none) vs. 0.8235 (weights) vs. 0.8221
(downsample); logistic regression 0.8192 / 0.8197 / 0.8196. But balancing made the predicted probabilities much
worse: Brier score 0.106 (none) vs. 0.174–0.180 (balanced), because the model then overestimates risk. Honest
probabilities matter for the UI ("your risk is X%"), and a lower threshold gives the same recall without
distorting them. This supersedes D-18 and the earlier plan (D-12) to balance; D-12's rule that the test set is
never balanced still holds.

### D-20 — CatBoost uses ordinal codes as numbers
**Date:** 2026-11-17
**Decision:** `GenHlth`, `Age`, `Education` and `Income` are given to CatBoost as ordered numbers, not as
categorical features.
**Why:** Result of `Modeling.ipynb` section 8 (one CV fold). Categorical mode was 8× slower (124 s vs. 16 s) for
a negligible gain (ROC-AUC 0.8227 vs. 0.8217). The codes have a natural order, which trees use well as numbers.
Logistic regression still gets them one-hot encoded, since it can't model non-linear effects otherwise.

### D-21 — Default settings in step 1
**Date:** 2026-11-17
**Decision:** Compare the models with sensible defaults (CatBoost: 500 trees, learning rate 0.1, depth 6;
logistic regression: default regularisation). No hyperparameter tuning in step 1.
**Why:** Equal, small tuning effort keeps the comparison fair. The expected gain from tuning on this data is small
(~0.01 ROC-AUC or less), and AutoML (step 2) will search settings automatically. Tuning of the final model can be
done in the training step if it is needed.

### D-22 — CatBoost is the manual winner
**Date:** 2026-11-17
**Decision:** CatBoost (no balancing, default settings) goes forward as the model AutoML must beat by more than
0.01 ROC-AUC (D-10). Logistic regression is not kept as a finalist.
**Why:** CatBoost was better on every metric. On the test set: ROC-AUC 0.825 vs. 0.819, PR-AUC 0.463 vs. 0.442,
precision at ~80% recall 33.0% vs. 32.4%. The gap (0.006) is inside the ~0.01 margin where D-10 allowed preferring
logistic regression for simplicity, but the team chose the better-performing model; explainability is handled
with CatBoost's feature importance.

### D-23 — Recall target 80%
**Date:** 2026-11-17
**Decision:** The decision threshold is set so the model catches **80% of people with prediabetes/diabetes**. The
threshold is chosen on cross-validated (out-of-fold) predictions, never on the test set.
**Why:** Balanced middle ground for screening. For CatBoost it gives ~33% precision (1 in 3 flagged people has
diabetes) and flags ~38% of people. On the test set: 79.6% recall, 33.0% precision at threshold 0.151. Going to
90% recall would flag half of everyone; 75% would miss 1 in 4. The exact threshold value is recomputed when the
final model is trained and saved together with it (D-15).
