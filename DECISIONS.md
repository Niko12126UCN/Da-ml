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
| D-24 | 2026-11-17 | 80/20 stratified train/test split + 5-fold CV on the training part | ✅ |
| D-25 | 2026-11-17 | Logistic regression preprocessing and settings | ✅ |
| D-26 | 2026-11-17 | Comparison metrics: ROC-AUC, PR-AUC, precision @ 80% recall, Brier score | ✅ |
| D-27 | 2026-11-17 | AutoML tool: AutoGluon 1.6.3 | ✅ |
| D-28 | 2026-11-17 | `da-ml-automl` environment and its versions (scikit-learn 1.7.2, PyTorch 2.13 CPU) | ✅ |
| D-29 | 2026-11-17 | AutoGluon settings: `best_quality`, 1 hour, optimise ROC-AUC | ✅ |
| D-30 | 2026-11-17 | Include neural networks in the AutoML run | ✅ |
| D-31 | 2026-11-17 | Run AutoML on the CPU, not the GPU | ✅ |
| D-32 | 2026-11-17 | Shared split file `data/splits/train_test_split.csv` | ✅ |
| D-33 | 2026-11-17 | How AutoGluon is compared with CatBoost | ✅ |
| D-34 | 2026-11-17 | Where model files are stored (not committed) | ✅ |
| D-35 | 2026-11-17 | Accept AutoGluon's sequential fold training (`ray` not installed) | ✅ |
| D-36 | 2026-11-17 | AutoML result: keep CatBoost as the model to deploy | ✅ |
| D-37 | 2026-11-18 | Design of the shared validation module `ml/validation.py` | ✅ |
| D-38 | 2026-11-18 | BMI: allowed range 12–98, decimals rounded; all other values must be whole numbers | ✅ |
| D-39 | 2026-11-18 | Only the three-class `Diabetes_012` file format is accepted for training | ✅ |
| D-40 | 2026-11-18 | Prediction input: values must be real numbers (no text, true/false or null) | ✅ |
| D-41 | 2026-11-18 | Training data must contain both people with and without diabetes | ✅ |
| D-42 | 2026-11-18 | Automated tests with pytest in `tests/`; flake8 with 120-character lines | ✅ |
| D-43 | 2026-11-18 | Training script `ml/train.py`: command line, own 80/20 split, same settings as step 1 | ✅ |
| D-44 | 2026-11-18 | Deployed model trained on the 80% training split; test metrics saved with it | ✅ |
| D-45 | 2026-11-18 | Model saved in CatBoost's own format (`model.cbm`) + `metadata.json` | ✅ |
| D-46 | 2026-11-18 | Each training run gets its own folder `models/<run_id>/`; `models/latest.txt` names the current one | ✅ |
| D-47 | 2026-11-18 | Small trained models are committed to git; the user does all commits | ✅ |
| D-48 | 2026-11-18 | Risk levels: low < screening threshold ≤ elevated < 30% ≤ high | ✅ |
| D-49 | 2026-11-18 | Messages recommend seeing a doctor, not a specific test | ✅ |
| D-50 | 2026-11-18 | API design: `/health`, `/model`, `/predict` (one person or a list, max 1,000) | ✅ |
| D-51 | 2026-11-18 | API runs on 127.0.0.1:5001, Flask debug mode off | ✅ |
| D-52 | 2026-11-18 | UI layout: question cards, tap-to-select buttons, result panel with risk bar | ✅ |
| D-53 | 2026-11-18 | UI styling: Bootstrap 5 + one small stylesheet, system fonts | ✅ |
| D-54 | 2026-11-18 | UI runs on 127.0.0.1:8050 and only talks to the API | ✅ |
| D-55 | 2026-11-18 | Questions in plain language; BMI calculated from height (cm) and weight (kg) | ✅ |
| D-56 | 2026-11-18 | Every question must be answered; nothing pre-filled except the "days" sliders (0) | ✅ |
| D-57 | 2026-11-18 | Very low risks shown as "less than 1%"; screening threshold shown with one decimal | ✅ |
| D-58 | 2026-11-18 | README with setup, run, train and test instructions | ✅ |
| D-59 | 2026-11-18 | Local deployment: `run.py` + `start.bat` start API and UI together on this PC | ✅ |
| D-60 | 2026-11-18 | Hosting (Docker / cloud) postponed; decided after local use | 🔶 |
| D-61 | 2026-11-18 | Quality check before a new model goes live; `--force` overrides | ✅ |
| D-62 | 2026-11-18 | The running app switches to a new live model by itself (no restart) | ✅ |
| D-63 | 2026-11-18 | Rejected models are kept (marked "not deployed"); unique folder names | ✅ |
| D-64 | 2026-11-18 | PyCharm run configuration "Train model" | ✅ |

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

### D-24 — 80/20 split + 5-fold cross-validation
**Date:** 2026-11-17 (applied in `Modeling.ipynb`; logged after the user asked about the split)
**Decision:** One stratified 80/20 train/test split with a fixed seed (`random_state=42`): 202,944 training rows and
50,736 test rows, both 15.8% positive. All comparisons use stratified 5-fold cross-validation on the training part.
The test set is used once, for the final evaluation. AutoML (step 2) uses the **same** split.
**Why:** 20% is already a large test set (~8,000 positive cases), so its scores are stable; the test results
matched CV within ~0.001 ROC-AUC. Cross-validation replaces a separate validation set, so a bigger hold-out (e.g.
70/30) would only take training data away without making the test more reliable. Stratifying keeps the real class
ratio in every part (D-12).

### D-25 — Logistic regression preprocessing and settings
**Date:** 2026-11-17 (applied in `Modeling.ipynb`; logged in the settings review after D-24)
**Decision:** scikit-learn pipeline with:
- `BMI`, `MentHlth`, `PhysHlth` → `StandardScaler`,
- `GenHlth`, `Age`, `Education`, `Income` → `OneHotEncoder(handle_unknown="ignore")`,
- the 14 binary 0/1 columns → passed through unchanged,
- `LogisticRegression(max_iter=2000)` with default regularisation (L2, `C=1.0`, `lbfgs` solver).
**Why:** Logistic regression is sensitive to feature scale, so the real-valued features are standardised. One-hot
encoding lets it learn a separate effect per level of the ordinal codes (e.g. risk rising with age, then dipping
at 80+), which a single linear weight can't capture. Binary flags are already on a 0/1 scale. `max_iter=2000`
avoids convergence warnings; the regularisation was left at its default per D-21.

### D-26 — Comparison metrics
**Date:** 2026-11-17 (applied in `Modeling.ipynb`; logged in the settings review after D-24)
**Decision:** Models are compared on four threshold-free metrics, computed on out-of-fold predictions:
**ROC-AUC** (main metric), **PR-AUC** (average precision), **precision at 80% recall**, and the **Brier score**
(accuracy of the predicted probabilities). Recall, precision, F1, accuracy and share flagged are reported at the
chosen threshold for the final test-set evaluation only.
**Why:** The threshold is decided after the comparison (D-23), so the comparison itself must not depend on one.
ROC-AUC and PR-AUC measure ranking; precision at 80% recall is the screening question in one number; the Brier score
was added because the UI will show probabilities, and it is what revealed that balancing distorts them (D-19).
Extends D-13.

### D-27 — AutoML tool: AutoGluon
**Date:** 2026-11-17
**Decision:** Use **AutoGluon** (`autogluon.tabular` 1.6.3) for the AutoML benchmark, not PyCaret.
**Why:** The question for step 2 is whether a clearly better model than CatBoost exists. AutoGluon is built for
the best possible score (bagging + stacked ensembles), so if it can't beat CatBoost by more than 0.01 ROC-AUC, we can
be confident CatBoost is close to the best this data allows. It is actively maintained, which matters if its model
gets deployed, and `leaderboard()` still shows every individual model. PyCaret would give a nicer comparison table,
but its development has slowed and it pins older package versions.

### D-28 — `da-ml-automl` environment
**Date:** 2026-11-17
**Decision:** Separate conda environment `da-ml-automl` from `environment-automl.yml` (D-11): Python 3.11,
`autogluon.tabular[catboost,lightgbm,xgboost,fastai]==1.6.3`, and the same versions as `da-ml` for numpy 1.26.4,
pandas 2.2.2, CatBoost 1.2.8 and matplotlib 3.8.2. Exceptions: **scikit-learn 1.7.2** (instead of 1.4.0) and
pyarrow/PyTorch as AutoGluon requires (PyTorch 2.13.0, CPU build).
**Why:** AutoGluon needs `pyarrow>=23`, which conflicts with `pyarrow==20.0.0` in `da-ml`. scikit-learn 1.4.0
ships an old Microsoft C++ runtime (`msvcp140.dll` 14.32); when it loads before PyTorch (CatBoost, XGBoost and
AutoGluon all import scikit-learn), PyTorch fails with `WinError 1114: c10.dll`. scikit-learn 1.7.2 fixes this at
the cause, so no import-order workaround is needed. The newer scikit-learn doesn't affect the comparison: the split
comes from a file (D-32), and CatBoost is the same version.

### D-29 — AutoGluon settings
**Date:** 2026-11-17
**Decision:** `TabularPredictor(label="diabetes", eval_metric="roc_auc")`, `fit(presets="best_quality",
time_limit=3600)`, `verbosity=1`. AutoGluon's own internal random seeds are left at their defaults. The test set is
not given to AutoGluon.
**Why:** `best_quality` (bagging + stacking) gives AutoGluon its best chance to beat CatBoost. 1 hour was chosen by
the team as enough for all model types on 200k rows; gains beyond that are usually very small. ROC-AUC is our
comparison metric (D-26), so AutoGluon optimises it directly.

### D-30 — Include neural networks
**Date:** 2026-11-17
**Decision:** Install the `fastai` extra (PyTorch), so AutoGluon also trains its two neural network models
(`NeuralNetTorch`, `NeuralNetFastAI`).
**Why:** Chosen by the team for a fairer benchmark that covers a really different type of model. They are usually
not better on tabular data like this, but now we can show it rather than assume it. Cost: ~1–2 GB larger install.
A 20k-row trial run on 2026-11-17 confirmed both neural networks train.

### D-31 — CPU, not GPU
**Date:** 2026-11-17
**Decision:** Run AutoGluon on the CPU (i7-13700H, 20 threads). PyTorch is the CPU build; the RTX 4050 laptop GPU is
not used.
**Why:** With a fixed time budget (D-29), a GPU wouldn't shorten the run, only fit more models into it. Only the
neural networks and CatBoost can use the GPU (XGBoost is installed as `xgboost-cpu`; LightGBM needs a special build;
random forest/extra trees/kNN are CPU only). The data (200k × 21) is too small for a GPU to pay off, and CatBoost on
the GPU uses different internal settings, which would weaken the comparison with our CPU CatBoost. Can be revisited
for large neural networks or tabular foundation models.

### D-32 — Shared split file
**Date:** 2026-11-17
**Decision:** `Modeling.ipynb` saves its 80/20 split (D-24) to `data/splits/train_test_split.csv` (`row` = row
number in the raw CSV, `split` = `train`/`test`). `AutoML.ipynb` reads this file instead of re-splitting, and
checks that it matches the dataset.
**Why:** The two notebooks run in different environments with different scikit-learn versions (D-28). Saving the
split guarantees both use exactly the same rows. After adding the cell, `Modeling.ipynb` was re-run and reproduced
all step 1 scores exactly.

### D-33 — How AutoGluon is compared with CatBoost
**Date:** 2026-11-17
**Decision:** In `AutoML.ipynb`:
- our CatBoost is **retrained in the same notebook** with the step 1 settings (D-20, D-21) on the same training
  rows, with its threshold from 5-fold out-of-fold predictions;
- two AutoGluon models are compared: its **best model** (usually an ensemble) and its **best single model** (no
  stacking or ensembling, easier to deploy), with thresholds from AutoGluon's out-of-fold predictions;
- all three are scored once on the test set with the same metrics (D-26) at 80% recall (D-23), plus prediction
  time and AutoGluon's model size;
- the D-10 rule is applied to the ROC-AUC difference;
- permutation feature importance of the best AutoGluon model on 5,000 test rows.
**Why:** Same data, same metrics and the same threshold method make the comparison fair. Including the best single
model shows what we'd get without the deployment cost of a full ensemble. Prediction time and size matter for
step 5 (API).

### D-34 — Model file storage
**Date:** 2026-11-17
**Decision:** AutoGluon saves its models to `models/autogluon_benchmark/`, which is in `.gitignore` (also
`AutogluonModels/` and `catboost_info/`). In `AutoML.ipynb`, CatBoost runs with `allow_writing_files=False`.
**Why:** AutoGluon's model folder can be several GB, and it can be recreated by re-running the notebook. Large
binary files don't belong in git. The final model for deployment is decided in a later step.

### D-35 — Sequential fold training (`ray` not installed)
**Date:** 2026-11-17 (noticed in the run log after the AutoML run)
**Decision:** Accept that AutoGluon trained the bagging folds **one after another** because the optional `ray`
package wasn't installed (it isn't part of the extras chosen in D-28). The run was not repeated with `ray`.
**Why:** AutoGluon still trained every model type, including both neural networks, within the hour; only a few
extra variants were skipped for lack of time. With `ray`, folds run in parallel and more variants fit in the same
budget, but every strong model already scored within 0.824–0.826, so more variants would not plausibly add the
+0.01 needed by D-10. Installing `ray` on Windows also adds another large, less-tested dependency. If AutoML is run
again, `ray>=2.43.0,<2.59.0` can be added to `environment-automl.yml`.

### D-36 — Keep CatBoost after the AutoML benchmark
**Date:** 2026-11-17
**Decision:** **CatBoost** (step 1 settings: D-19 to D-21) remains the model to deploy. The AutoGluon models are not
used further; they stay in the git-ignored `models/autogluon_benchmark/` folder only as a record.
**Why:** Applying the rule agreed before the run (D-10) to the test set (`AutoML.ipynb`, run 11:04–12:08):

| Model | ROC-AUC | Precision @ ~80% recall | Predict 50k rows | Size |
|---|---|---|---|---|
| AutoGluon best (WeightedEnsemble_L2) | 0.8261 | 33.3% | 8.8 s | ~3 GB (all models) |
| AutoGluon best single (bagged CatBoost) | 0.8260 | 33.2% | 0.12 s | part of the above |
| **Our CatBoost** | 0.8248 | 33.1% | 0.01 s | a few MB |

The best gain is +0.0012 ROC-AUC, far below the +0.01 threshold. All strong model types (CatBoost, LightGBM,
XGBoost, two neural networks) scored 0.824–0.826, which shows the data limits the score, not the model. The
ensemble's tiny gain doesn't justify ~700× slower predictions, ~3 GB of files and AutoGluon + PyTorch as API
dependencies. Our CatBoost retrained in `AutoML.ipynb` scored 0.8248 vs. 0.8247 in `Modeling.ipynb`; the 0.0001
difference comes from the different scikit-learn version and thread scheduling and doesn't affect the conclusion.
Possible later improvement: bag our CatBoost over folds like AutoGluon did (~+0.001), only if it is ever needed.

### D-37 — Design of `ml/validation.py`
**Date:** 2026-11-18
**Decision:** One module defines the "exact format" and is used by both the training script and the API (D-14):
- `FEATURE_RANGES` is the single definition of the 21 features and their allowed (min, max) values, in the order
  of the raw CSV; `Diabetes_012` must be 0–2.
- `validate_training_data(df)` checks a whole dataset; `validate_features(data)` checks prediction input: one
  person (dict), several people (list of dicts) or a DataFrame.
- Checked: missing / unexpected / duplicate columns, empty input, missing values, non-numeric values, non-whole
  numbers, values out of range.
- Column **order** doesn't matter; **extra** columns are rejected (not silently dropped).
- **All** problems are reported at once in a `ValidationError` (`.errors` is a list of readable messages). Only
  exception: if columns are missing or unexpected, the values aren't checked until the structure is fixed.
- Output: a clean integer DataFrame in the canonical column order, so the model always gets the same layout.
- Duplicate **rows** are not checked, since they are kept on purpose (D-17).
**Why:** A single definition can't drift apart between training and API. Reporting all problems at once makes a bad
file or request quick to fix. Rejecting extra columns catches wrong files (e.g. a different dataset) instead of
training on the wrong data.

### D-38 — BMI range and rounding; whole numbers elsewhere
**Date:** 2026-11-18
**Decision:** BMI must be between **12 and 98**, the range seen in the dataset. BMI may be sent with decimals and is
**rounded** to a whole number, like in the dataset. All other features must be whole numbers; `9.0` is accepted as
9, but `9.5` is rejected.
**Why:** The model has only seen BMI values 12–98; predictions outside that range would be guesses. Users (or the UI)
may calculate BMI from height and weight, which gives decimals; rounding matches how the training data stores BMI.
For coded answers (age group, yes/no…), a decimal means the input is wrong, so it is rejected rather than guessed.

### D-39 — Only the three-class file format
**Date:** 2026-11-18
**Decision:** Training data must have the `Diabetes_012` column (format of
`diabetes_012_health_indicators_BRFSS2015.csv`). The two Kaggle files with `Diabetes_binary` are rejected with
"missing columns: Diabetes_012 / unexpected columns: Diabetes_binary".
**Why:** That is the "exact format" the pipeline is built for (D-02), and the yes/no target is derived from it by the
pipeline itself (D-08). The binary files lose the prediabetes/diabetes distinction, and the 50/50 file has a
different class ratio, which would distort the threshold and probabilities (D-19). Can be extended later if needed.

### D-40 — Strict types in prediction input
**Date:** 2026-11-18
**Decision:** In `validate_features` with dicts (the JSON the API will receive), every feature value must be a real
number. Text (`"31"`), `true`/`false` and `null` are rejected, even if they could be converted.
**Why:** The API should get exactly what the UI sends; a string or boolean means a bug in the caller, and silently
converting it would hide that bug. CSV files (training) are read by pandas, so there numbers stored as text are fine.

### D-41 — Both classes in the training data
**Date:** 2026-11-18
**Decision:** `validate_training_data` rejects a dataset where `Diabetes_012` is only 0 or only 1/2.
**Why:** A model can't be trained, and no recall threshold can be chosen, without both groups.

### D-42 — Tests and style checks
**Date:** 2026-11-18
**Decision:** Automated tests with **pytest** in `tests/` (`tests/test_validation.py`, run from the project root with
`python -m pytest tests`). Code is checked with **flake8** using `--max-line-length 120`. `ml/` is a Python package
(`ml/__init__.py`), so the API can `from ml.validation import ...`.
**Why:** pytest and flake8 are already in `environment.yml` (as in the taxi project). The tests prove that the real
Kaggle file passes unchanged and that each kind of bad input is rejected with a clear message, and they guard the
validation when it is changed later. 120 characters matches the line length already used in the notebooks.

### D-43 — Training script `ml/train.py`
**Date:** 2026-11-18
**Decision:** A command-line script, run from the project root in `da-ml`:
`python -m ml.train [--data <csv>] [--out <folder>]` (defaults: the Kaggle CSV and `models/`). It:
1. reads and validates the CSV (`validate_training_data`, D-37),
2. builds the yes/no target (D-08) and makes its **own** 80/20 stratified split with seed 42 (D-24),
3. runs 5-fold stratified CV (seed 42) on the training rows and picks the threshold for **80% recall** from the
   out-of-fold predictions (D-23),
4. trains CatBoost with the step 1 settings (500 trees, learning rate 0.1, depth 6, ordinal codes as numbers, no
   balancing; D-19 to D-21) on all training rows,
5. evaluates once on the test rows with the same metrics as the notebooks (D-26), and saves the result (D-45, D-46).
All settings are constants at the top of the file, each with its decision ID.
**Why:** The pipeline must be able to "take a dataset in this exact format and train a model" without a notebook.
The script makes its own split instead of reading `data/splits/train_test_split.csv`, because that file only fits
the Kaggle CSV and the pipeline should work for any CSV in the format. For the Kaggle CSV the split is identical
(checked by `tests/test_train.py`), and the script reproduces `Modeling.ipynb` exactly (see `RESULTS.md`, step 4).

### D-44 — Deployed model trained on the training split
**Date:** 2026-11-18
**Decision:** The model that gets deployed is the one trained on the **80% training rows**. It is not retrained on
all data. Its test-set metrics are saved in `metadata.json`.
**Why:** Chosen by the team: every deployed model comes with an honest score measured on data it has never seen.
Retraining on 100% would add 20% more data for a likely gain of ~0.001 ROC-AUC, but the deployed model itself would
have no untouched test score.

### D-45 — Model format
**Date:** 2026-11-18
**Decision:** The model is saved with CatBoost's own `save_model` as `model.cbm` (~0.6 MB), next to a
`metadata.json` with: run ID and time, purpose ("not a diagnosis"), the feature list and allowed ranges, the
threshold and recall target, CV and test metrics, row counts, settings, the data file's name and SHA-256
fingerprint, and the Python / CatBoost / scikit-learn / pandas / numpy versions.
**Why:** CatBoost's own format loads with just `catboost` (no pickle, so no risk of running arbitrary code when
loading, and no dependency on the exact Python objects). The metadata gives the API everything it needs (threshold,
features) and makes each model traceable: which data, which settings, which versions, how good.

### D-46 — One folder per training run
**Date:** 2026-11-18
**Decision:** Each run creates `models/catboost_<YYYYMMDD>_<HHMMSS>/` and then writes that folder's name to
`models/latest.txt`. The API will load the model named in `latest.txt`. Existing runs are never overwritten.
**Why:** Old models stay available, so a bad new model can be rolled back by editing one line. The version history
matches the plan's "saved model, metrics and threshold (versioned)" (D-15).

### D-47 — Trained models in git; user commits
**Date:** 2026-11-18
**Decision:** Trained model folders (`model.cbm` + `metadata.json`) and `models/latest.txt` are committed to git as
long as they are small (~0.6 MB per run). Large artifacts stay git-ignored (AutoGluon, D-34). **Claude never runs
`git commit`;** the user commits after each step, using the suggested commit message.
**Why:** The user's decision: with a model in the repo, the API and UI work straight after cloning, without
training first. The user commits themselves to stay in control of what ends up in the repo. Old runs that are no
longer needed can be deleted.

### D-48 — Risk levels
**Date:** 2026-11-18
**Decision:** The predicted probability is shown as one of three levels (defined in `ml/risk.py`):
- **low:** below the model's screening threshold (from `metadata.json`; 15.1% for `catboost_20261118_084406`),
- **elevated:** from the threshold up to 30%,
- **high:** 30% and above (`HIGH_RISK = 0.30`).
Elevated and high are both "flagged". The screening threshold stays tied to the model (re-chosen for 80% recall on
every training run, D-23); the 30% cut-off is fixed.
**Why:** Chosen by the team after reviewing the out-of-fold table (`RESULTS.md`, step 5). 30% is about twice the
average risk (15.8%), which is easy to explain. About 1 in 5 people are "high", and 44.5% of them have
prediabetes/diabetes, ~9× the low group (5.1%). The "high" cut-off doesn't change who is flagged, so it doesn't
change false positives or false negatives; it only changes how urgent the message is. The probabilities are
calibrated, so the percentage can be shown as is.

### D-49 — Messages recommend a doctor, not a test
**Date:** 2026-11-18
**Decision:** The messages tell the user to **see/talk to their doctor**, never to take a specific test (e.g. blood
sugar / HbA1c):
- low: "Your estimated risk is low (X%). If you have symptoms or concerns, talk to your doctor."
- elevated: "Your estimated risk is above average (X%). Consider talking to your doctor about your diabetes risk."
- high: "Your estimated risk is high (X%), about twice the average or more. We recommend that you see your doctor
  about your diabetes risk."
Every result also carries: "This is a screening estimate based on survey answers, not a diagnosis."
**Why:** The team's decision: the tool should send people to a doctor, who decides what to check; recommending a
specific test is medical advice the tool shouldn't give. The low message still mentions a doctor because a low
estimate doesn't rule out diabetes (20% of diabetics are in the low group). Replaces the example wording
("consider a blood glucose / HbA1c test") in `PROJECT_PLAN.md` section 7. The purpose in D-07 is unchanged.

### D-50 — API design
**Date:** 2026-11-18
**Decision:** Flask app in `api/main.py` (like the taxi project), created by `create_app()`:
- `GET /health` → status and the loaded model's run ID,
- `GET /model` → feature list, allowed ranges, threshold, risk-level cut-offs, the model's test metrics and the
  disclaimer (so the UI doesn't need to hard-code them),
- `POST /predict` → JSON object (one person) or JSON list (several people, at most 1,000 per request). Each result
  has `probability` (rounded to 4 decimals), `risk_level`, `flagged`, `message` and `disclaimer`, plus the model's
  run ID.
- Input is checked with `validate_features` (D-37, D-40); invalid input returns **HTTP 400** with the full list of
  problems as `errors`. A body that isn't JSON also returns 400.
- At start-up the API loads the model named in `models/latest.txt` (D-46) and refuses to start if the model's
  feature list differs from `ml/validation.py`.
- Tests in `tests/test_api.py` use Flask's test client on the committed model; `tests/` is a package
  (`tests/__init__.py`) so the test files can share example inputs.
**Why:** The endpoints cover what the UI needs (step 6) and keep all model knowledge on the API side. Accepting a
list allows batch screening; the 1,000 limit keeps a single request from using too much memory or time. The
start-up check catches a model trained for a different format before it gives wrong answers.

### D-51 — Host, port and debug mode
**Date:** 2026-11-18
**Decision:** `python -m api.main` serves on **127.0.0.1:5001** (only reachable from the same computer), with Flask
debug mode **off**. Flask's built-in server is used for now; a production server is chosen in the deployment step.
**Why:** Same port as the taxi project's API. Binding to 127.0.0.1 keeps the API private until deployment is
decided. Debug mode is off because Flask's debugger lets anyone who can reach it run code on the machine, and it
isn't needed (errors go to the log). Note for deployment: `gunicorn` (in `environment.yml`) doesn't run on Windows,
so Windows deployment would need another server (e.g. `waitress`) or Docker.

### D-52 — UI layout
**Date:** 2026-11-18
**Decision:** One Dash page (`ui/app.py`, Dash + dash-bootstrap-components, like the taxi project):
- the 21 questions grouped on four cards: **About you**, **Your health**, **Lifestyle**, **Healthcare**,
- choices as **tap-to-select buttons** (pill-shaped), not radio buttons; Yes/No questions show the question and the
  buttons on one line on wide screens,
- dropdowns for age group, education and income; sliders (0–30, value shown as "X days") for the two "days"
  questions,
- a **result panel** next to the questions (it stays in view while scrolling on wide screens, and sits below them on
  phones) with the percentage, a coloured level badge, a **risk bar** with three equal-width zones (low / elevated /
  high) and a marker, the message, "Of 100 people with answers like yours, about X have prediabetes or diabetes",
  and the disclaimer,
- problems (unanswered questions, impossible height/weight, API not running) are shown in the result panel as a
  list in plain language.
**Why:** The team asked for something simple and clean that doesn't look like an old-school form page. Buttons are
quicker to use and easier to read than small radio buttons; one line per Yes/No question roughly halves the page
length. The risk bar's zones are equal width (instead of 0–100% to scale) so the low and elevated zones, which
are only 15 percentage points wide, stay readable; the level is also written out, so the colour is never the only
signal.

### D-53 — UI styling
**Date:** 2026-11-18
**Decision:** Bootstrap 5 (`dbc.themes.BOOTSTRAP`, loaded from the jsDelivr CDN) plus one stylesheet,
`ui/assets/style.css`: light grey background, white cards with rounded corners and a thin border, one blue accent
colour, green / amber / red for low / elevated / high, and the operating system's own fonts. The page has a viewport
tag so it renders at phone width on phones.
**Why:** Clean and modern with very little custom code. System fonts need no download and look native on every
device. Note: the Bootstrap CSS comes from a CDN, so the browser needs internet access; for an offline deployment
the file can be put in `ui/assets/` instead.

### D-54 — UI host, port and API connection
**Date:** 2026-11-18
**Decision:** `python -m ui.app` serves on **127.0.0.1:8050** with debug mode off. It calls the API at
`http://127.0.0.1:5001`, which can be changed with the environment variable `DA_ML_API_URL`. The UI contains **no
model logic**: it sends the answers to `/predict` and reads the threshold and "high" cut-off from `/model` (cached
after the first call).
**Why:** Port 8050 matches the taxi project's UI. Keeping all model knowledge in the API means a new model (or new
cut-offs) only needs an API restart, and the UI can't show different cut-offs from the ones the API uses. The
environment variable lets the API live elsewhere in a deployment. Debug mode off for the same reason as the API
(D-51).

### D-55 — Plain-language questions; BMI from height and weight
**Date:** 2026-11-18
**Decision:** The questions are worded like the original BRFSS 2015 survey questions in plain language (for example
"Has a doctor ever told you that you have high blood pressure?"), and the codes are never shown. BMI is calculated
from **height in cm and weight in kg**, rounded to one decimal and sent to the API, which rounds it to a whole
number like the training data (D-38). Income is asked in **US dollars**, as in the survey.
**Why:** The model learned from people answering those questions, so close wording keeps the answers comparable.
Few people know their BMI, but everyone knows their height and weight. Metric units are the standard almost everywhere outside the US;
the BMI is the same either way. Income stays in US dollars because the model's income groups are
2015 US dollar ranges; converting them would suggest a precision the data doesn't have.

### D-56 — All questions required
**Date:** 2026-11-18
**Decision:** Every question starts unanswered and must be answered before a result is shown; missing answers are
listed by name. Only the two "days" sliders start at 0, which is itself a valid answer ("no bad days").
**Why:** Pre-filling "No" would quietly bias the result for anyone who skips a question. The model needs all 21
values, and guessing missing ones would make the percentage less honest.

### D-57 — "Less than 1%" and the threshold with one decimal
**Date:** 2026-11-18
**Decision:** Probabilities below 0.5% are shown as **"less than 1%"** (message, from `ml/risk.py`) / **"<1%"** (big
number) and "fewer than 1 has prediabetes or diabetes", instead of "0%". On the risk bar the screening threshold is
shown with one decimal (**15.1%**).
**Why:** Found in the screenshots: "0%" and "about 0 of 100" suggest zero risk, which the model can't claim. With
the threshold rounded to "15%", a person at 15.0% would be "low" right next to a "15%" mark, which looks wrong.

### D-58 — README
**Date:** 2026-11-18
**Decision:** `README.md` explains what the project is, links to the documents, and gives the commands to set up
the environment, run the API and UI, train a new model and run the tests.
**Why:** With two servers to start, someone cloning the repository needs the exact commands. The README was empty.

### D-59 — Local deployment with `run.py` and `start.bat`
**Date:** 2026-11-18
**Decision:** For now the app runs **on the user's PC**. `run.py` (project root) starts the API (`python -m api.main`)
and the UI (`python -m ui.app`) as two child processes with the same Python, waits until both answer (API
`/health`, UI `/`, up to 60 s each), opens the browser at http://127.0.0.1:8050, and stops both on Ctrl+C. If one
server stops unexpectedly, the other is stopped too and the script exits with an error. Before starting, it checks
that ports 5001 and 8050 are free ("Is the app already running?"). Options: `--no-browser`, and `--check` (start,
check, stop; exit code 0 = OK), which `tests/test_run.py` uses. `start.bat` can be double-clicked: it looks for the
`da-ml` environment in `%USERPROFILE%\miniconda3\envs\da-ml` (or `anaconda3`), runs `run.py`, and keeps the window
open if something goes wrong. Flask's and Dash's built-in servers are used, bound to 127.0.0.1.
**Why:** The team's decision: run it locally first, decide on hosting afterwards (D-60). One command (or a
double-click) instead of two terminals makes it easy to use. The built-in servers are fine for one user on one PC;
their "not for production" warning matters once others can reach the app, which is part of the hosting decision
(note D-51: `gunicorn` doesn't run on Windows). `start.bat` uses the environment's `python.exe` directly because
`conda` isn't on the PATH on this PC.

### D-60 — Hosting postponed 🔶
**Date:** 2026-11-18
**Decision:** Where to host the app for others (Docker container, a cloud service such as Azure or Render, or
something else) is decided **later**, after the app has been used locally.
**Why:** The team's decision. Things to settle then: a production web server instead of the built-in ones (D-51,
D-59), whether the API and UI run as one or two services, the UI's API address (`DA_ML_API_URL`, D-54), serving the
Bootstrap CSS without a CDN if needed (D-53), HTTPS, and the limits in `MODEL_EXPLAINED.md` (population, EU medical
device rules) if the public can use it.

### D-61 — Quality check before a new model goes live
**Date:** 2026-11-18
**Decision:** After training, `python -m ml.train` only makes the new model live if it passes all of:
- test ROC-AUC **≥ 0.80** (`MIN_ROC_AUC`),
- test ROC-AUC **not more than 0.01 below the live model** (`MAX_ROC_AUC_DROP`, compared with the model named in
  `models/latest.txt`; skipped if there is none),
- recall at the chosen threshold on the test split **≥ 75%** (`MIN_RECALL`; the target is 80%, D-23).
If it fails, the new model is still saved, but the live model stays, and the script prints the failed checks and
exits with code **2** (0 = live, 1 = error). `--force` makes it live anyway. The result (`passed`, `failures`,
`forced`, `compared_with`, `deployed`) is stored in the model's `metadata.json` under `quality_check`.
**Why:** The team wants a full pipeline where training a new model deploys it automatically, and chose to deploy
**only if it passes a check**. Without one, a bad dataset (e.g. answers coded differently but still within the
allowed ranges, which validation can't catch) would silently put a worse model live. 0.01 is the same margin as the
AutoML rule (D-10); it allows normal variation between datasets and test splits. 0.80 is just below the 0.819–0.826
every reasonable model reached (`RESULTS.md`). 75% recall leaves room around the 80% target, because the threshold
is chosen on cross-validation and lands at ~79.5% on the test split. Exit code 2 lets a script or scheduler see that
nothing was deployed.

### D-62 — Automatic switch in the running app
**Date:** 2026-11-18
**Decision:**
- **Training** writes `models/latest.txt` only after the new model folder is completely saved, and replaces it in
  one atomic step (`ml.train.activate`: write `latest.tmp`, then `os.replace`), so the API never reads a
  half-written name.
- **The API** (`ModelStore` in `api/main.py`) reads `models/latest.txt` on every request. When it names a different
  model, the API loads it and switches to it; requests during loading still use the current model. If the new model
  can't be loaded (missing or broken files), the current model keeps serving, the error is logged, and that model
  isn't retried on every request. The API still refuses to **start** without a usable model.
- **The UI** caches the threshold and "high" cut-off **per model** and fetches them again from `/model` when
  `/predict` answers with a different model ID.
This extends D-46 (`latest.txt` names the live model) and D-50 (the API loaded the model only at start-up).
**Why:** The team's goal: "when I train a new model, everything just deploys in switch to that new model". With
this, `python -m ml.train` is the only step; the app doesn't have to be restarted. Reading a one-line file per
request costs microseconds. Keeping the old model when the new one is broken means a bad deployment can't take the
app down. Checked on 2026-11-18 with the app running (`RESULTS.md`, step 8).

### D-63 — Rejected models are kept; unique folder names
**Date:** 2026-11-18
**Decision:** A model that fails the quality check is still saved in `models/<run_id>/`, with
`"deployed": false` in its metadata. If two runs start in the same second, the second folder gets a suffix (`_2`,
`_3` …) instead of failing.
**Why:** Keeping rejected models makes it possible to see *why* a run was rejected, and to put it live later after
all, by writing its folder name into `latest.txt`, if the team decides the check was too strict for that data. Folders are ~0.6 MB; rejected ones can be deleted (or not
committed, D-47) when no longer needed. Same-second runs can happen in scripted pipelines.

### D-64 — PyCharm run configuration "Train model"
**Date:** 2026-11-18
**Decision:** `.idea/runConfigurations/Train_model.xml` adds **"Train model"** to PyCharm's run menu. It runs
`python -m ml.train --data "$PROJECT_DIR$/data/raw_data/diabetes_012_health_indicators_BRFSS2015.csv"` from the
project root with the project interpreter (`da-ml`). To train on another file, change the path under **Run → Edit
Configurations… → Train model → Parameters**.
**Why:** The team asked for a PyCharm command for now, pointing at the same file used so far. Run configurations in
`.idea/runConfigurations/` can be committed, so the command comes with the repository. The path is written out
(even though it's the script's default) so it's obvious where to change it.
