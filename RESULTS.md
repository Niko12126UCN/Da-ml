# Results

All **measured** findings of the project, step by step. For *what was decided because of them* (and why), see
[`DECISIONS.md`](DECISIONS.md); for the plan, see [`PROJECT_PLAN.md`](PROJECT_PLAN.md); for the expectations
before any model was trained, see [`Diabetes_Prediction_Feasibility.md`](Diabetes_Prediction_Feasibility.md).
For how a prediction is made (with a worked example) and the limits of the model, see
[`MODEL_EXPLAINED.md`](MODEL_EXPLAINED.md).

**Common setup for all results** (unless stated otherwise):

| Setting | Value | Decision |
|---|---|---|
| Data | `diabetes_012_health_indicators_BRFSS2015.csv`, 253,680 rows, 21 features | D-02 |
| Target | 1 = prediabetes or diabetes (15.8% of rows), 0 = no diabetes | D-08 |
| Split | 80/20 stratified, seed 42: 202,944 train / 50,736 test, both 15.8% positive | D-24, D-32 |
| Comparisons | 5-fold stratified cross-validation (CV) on the training rows | D-24 |
| Final check | Test set, used once per model | D-24 |
| Threshold | Set for 80% recall on out-of-fold predictions, then applied to the test set | D-23 |

**Metrics:** ROC-AUC (ranking, 0.5 = random, 1 = perfect) · PR-AUC (ranking on the rare class, random ≈ 0.16) ·
precision @ 80% recall (share of flagged people who really have diabetes) · Brier score (error of the predicted
probabilities, lower is better). See D-26.

---

## Step 0 — EDA (`ml/notebooks/EDA.ipynb`, 2026-11-16)

- No missing values; all coded columns within their allowed ranges; BMI 12–98.
- 23,899 duplicate rows according to pandas (the profiling report counts 11,187 duplicated row patterns).
- 1,834 identical answer patterns appear with **different** labels, so some people can't be told apart from the
  features at all.
- Strongest single relationships with the target (Spearman): `GenHlth` 0.30, `HighBP` 0.27, `BMI` 0.24,
  `DiffWalk` 0.22, `HighChol` 0.21, `Age` 0.19. All other features are below 0.18.

## Step 1 — Model comparison (`ml/notebooks/Modeling.ipynb`, 2026-11-17)

### Models × balancing (5-fold CV on the training data)

| Model | Balancing | ROC-AUC | PR-AUC | Precision @ 80% recall | Brier |
|---|---|---|---|---|---|
| Logistic regression | none | 0.8192 | 0.4388 | 0.3243 | 0.1076 |
| Logistic regression | class weights | 0.8197 | 0.4368 | 0.3259 | 0.1783 |
| Logistic regression | 50/50 downsampling | 0.8196 | 0.4370 | 0.3254 | 0.1785 |
| CatBoost | **none** | **0.8244** | 0.4546 | **0.3300** | **0.1062** |
| CatBoost | class weights | 0.8235 | **0.4573** | 0.3280 | 0.1739 |
| CatBoost | 50/50 downsampling | 0.8221 | 0.4459 | 0.3274 | 0.1795 |

**Finding:** balancing doesn't improve the ranking, but it makes the probabilities much worse (Brier 0.17–0.18 vs.
0.106): the model overestimates everyone's risk. → **No balancing** (D-19).

### Duplicates (CatBoost, no balancing, 5-fold CV)

| Variant | ROC-AUC | PR-AUC | Precision @ 80% recall | Brier |
|---|---|---|---|---|
| Standard CV (duplicates kept) | 0.8244 | 0.4546 | 0.3300 | 0.1062 |
| Grouped CV (identical rows never split between train and validation) | 0.8248 | 0.4564 | 0.3297 | 0.1061 |
| Duplicates removed from training | 0.8238 | 0.4537 | 0.3291 | 0.1063 |

**Finding:** differences below 0.001. Duplicates neither inflate the scores nor hurt the model. → **Keep** (D-17).

### CatBoost: ordinal codes as numbers vs. categories (one CV fold)

| Variant | ROC-AUC | PR-AUC | Precision @ 80% recall | Brier | Training time |
|---|---|---|---|---|---|
| Numeric (ordered numbers) | 0.8217 | 0.4524 | 0.3257 | 0.1067 | 9–16 s |
| Categorical | 0.8227 | 0.4552 | 0.3266 | 0.1065 | 98–124 s |

Training times varied between the two runs of the notebook; categorical mode was 8–11× slower each time.
**Finding:** +0.001 ROC-AUC is not worth ~10× the training time. → **Numbers** (D-20).

### Threshold table (5-fold out-of-fold predictions, no balancing)

What we get at different recall targets: precision = share of flagged people who really have diabetes; flagged =
share of all people sent for a test.

| Recall target | Logistic regression: threshold | precision | flagged | CatBoost: threshold | precision | flagged |
|---|---|---|---|---|---|---|
| 60% | 0.247 | 40.5% | 23.3% | 0.262 | 41.6% | 22.7% |
| 70% | 0.195 | 36.7% | 30.1% | 0.206 | 37.2% | 29.6% |
| 75% | 0.168 | 34.5% | 34.3% | 0.179 | 35.2% | 33.6% |
| **80%** | 0.143 | 32.4% | 38.9% | **0.151** | **33.0%** | **38.2%** |
| 85% | 0.116 | 29.9% | 44.7% | 0.122 | 30.6% | 43.7% |
| 90% | 0.089 | 27.3% | 51.9% | 0.089 | 27.7% | 51.1% |

→ **80% recall** chosen by the team (D-23).

### Final test-set evaluation (trained on all training rows, no balancing)

| Model | ROC-AUC | PR-AUC | Brier | Threshold | Recall | Precision | F1 | Accuracy | Flagged |
|---|---|---|---|---|---|---|---|---|---|
| Dummy ("no" for everyone) | 0.500 | – | – | – | 0% | – | – | 84.2% | 0% |
| Logistic regression | 0.8190 | 0.4421 | 0.1075 | 0.143 | 79.4% | 32.4% | 0.460 | 70.6% | 38.7% |
| **CatBoost** | **0.8247** | **0.4633** | **0.1057** | 0.151 | 79.6% | **33.0%** | **0.467** | 71.3% | 38.0% |

**Findings:** both models clearly beat the dummy; CatBoost is slightly better on every metric. Test scores match
CV within ~0.001, so the split is not a lucky one. → **CatBoost** (D-22).

## Step 2 — AutoML benchmark (`ml/notebooks/AutoML.ipynb`, 2026-11-17, 11:04–12:08)

AutoGluon 1.6.3, `best_quality`, 1 hour, CPU, neural networks included (D-27 to D-31). Training took 60.3 minutes.
The folds were trained one after another because `ray` was not installed (D-35), and one model variant was skipped
for lack of time. No second stacking layer was trained; the best model is a weighted ensemble of the layer 1 models.

### AutoGluon leaderboard (all models)

`score_val` = AutoGluon's own out-of-fold ROC-AUC on the training rows; `score_test` = ROC-AUC on our test set.
`_BAG_L1` = bagged model (8 folds) in layer 1.

| # | Model | Test ROC-AUC | Validation ROC-AUC | Training time | Predict test set |
|---|---|---|---|---|---|
| 1 | WeightedEnsemble_L2 | **0.8261** | 0.8266 | 2,581 s (all its models) | 9.1 s |
| 2 | CatBoost_BAG_L1 | 0.8260 | 0.8261 | 163 s | 0.12 s |
| 3 | NeuralNetFastAI_BAG_L1 | 0.8254 | 0.8248 | 1,522 s | 4.7 s |
| 4 | LightGBMXT_BAG_L1 | 0.8251 | 0.8255 | 20 s | 1.1 s |
| 5 | LightGBM_BAG_L1 | 0.8248 | 0.8250 | 12 s | 0.7 s |
| 6 | XGBoost_BAG_L1 | 0.8246 | 0.8250 | 10 s | 0.5 s |
| 7 | NeuralNetTorch_BAG_L1 | 0.8244 | 0.8239 | 831 s | 1.2 s |
| 8 | LightGBMLarge_BAG_L1 | 0.8244 | 0.8249 | 24 s | 1.4 s |
| 9 | CatBoost_r177_BAG_L1 | 0.8208 | 0.8190 | 10 s | 0.5 s |
| 10 | RandomForestGini_BAG_L1 | 0.8184 | 0.8177 | 7 s | 1.2 s |
| 11 | RandomForestEntr_BAG_L1 | 0.8179 | 0.8175 | 7 s | 1.1 s |
| 12 | ExtraTreesEntr_BAG_L1 | 0.8143 | 0.8139 | 6 s | 1.0 s |
| 13 | ExtraTreesGini_BAG_L1 | 0.8142 | 0.8138 | 6 s | 1.0 s |

### Head-to-head on the test set (threshold for 80% recall from out-of-fold predictions)

| Model | ROC-AUC | PR-AUC | Precision @ 80% recall | Brier | Threshold | Recall | Precision | F1 | Accuracy | Flagged | Predict 50k rows |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AutoGluon best (WeightedEnsemble_L2) | 0.8261 | 0.4671 | 0.3296 | 0.1054 | 0.154 | 79.5% | 33.3% | 0.469 | 71.7% | 37.6% | 8.76 s |
| AutoGluon best single (CatBoost_BAG_L1) | 0.8260 | 0.4669 | 0.3298 | 0.1054 | 0.154 | 79.5% | 33.2% | 0.469 | 71.6% | 37.7% | 0.12 s |
| **Our CatBoost** (step 1 settings) | 0.8248 | 0.4636 | 0.3283 | 0.1057 | 0.151 | 79.5% | 33.1% | 0.467 | 71.4% | 37.9% | 0.01 s |

**The same results in people** (test set: 50,736 people, 7,995 with prediabetes/diabetes). Recall is ~79.5% for all
three **by design**, because each threshold was set for 80% recall (D-23); the difference is how many false alarms
each model needs to get there. Counts are derived from the rounded recall / share-flagged values, so they can be off
by a few people.

| Model | Caught | Missed | Flagged in total | False alarms | Precision |
|---|---|---|---|---|---|
| AutoGluon best (ensemble) | 6,358 | 1,637 | 19,097 | 12,739 | 33.3% |
| AutoGluon best single | 6,354 | 1,641 | 19,127 | 12,774 | 33.2% |
| Our CatBoost | 6,358 | 1,637 | 19,234 | 12,876 | 33.1% |

The ensemble needs ~140 fewer false alarms out of 50,736 people (0.3%), about 1 fewer unnecessary test per 360 people
screened. Accuracy is ~71.5% for all three, lower than the dummy's 84.2% because many false alarms are accepted
in order to catch 80% of diabetics. For every flagged person, the chance of actually having prediabetes or diabetes
is about 1 in 3, whichever model is used.

AutoGluon model folder: ~2,975 MB (all models). Our CatBoost retrained here scored 0.8248 vs. 0.8247 in
`Modeling.ipynb` (different scikit-learn version / thread scheduling, D-36).

### Findings

1. **Gain over our CatBoost: +0.0012 ROC-AUC**, far below the +0.01 required by D-10. → **Keep CatBoost** (D-36).
2. At the 80% recall threshold the three models are practically identical (precision 33.1–33.3%).
3. **All strong model types score 0.824–0.826** (CatBoost, LightGBM, XGBoost, both neural networks); random forest
   and extra trees are slightly behind (0.814–0.818). The data limits the score, not the choice of model.
4. The neural networks were by far the slowest to train (14–25 minutes) and no better than the boosted trees.
5. The ensemble costs ~700× slower predictions and ~3 GB of files for its +0.001.
6. AutoGluon's best single model is a **bagged CatBoost** (8 CatBoost models averaged); that is where most of its
   small gain comes from.

## Step 3 — Input validation (`ml/validation.py`, 2026-11-18)

`python -m pytest tests` in the `da-ml` environment: **30 tests passed** (`tests/test_validation.py`); flake8 clean.

- The real Kaggle file (253,680 rows) passes, and the output has exactly the same values as the raw CSV, converted
  to integers.
- The binary Kaggle file format is rejected (D-39).
- Rejected with a clear message: missing / unexpected columns, missing values, non-numeric values, decimals in coded
  columns, out-of-range values (e.g. Age 14, BMI 5 or 120, MentHlth 31, HighBP 2, Income 9), a target of 3, a target
  with only one class, empty input, wrong input shapes, and text / true / null in prediction input.
- Several problems in one input are all reported together, also when one value has the wrong type.

## Step 4 — Training script (`ml/train.py`, 2026-11-18)

`python -m pytest tests`: **33 tests passed** (30 validation + 3 training); flake8 clean.

`python -m ml.train` on the Kaggle CSV (1 min 21 s in total; ~12 s per CV fold, 14 s for the final model) →
`models/catboost_20261118_084406/` (`model.cbm` 563 KB + `metadata.json`):

| | ROC-AUC | PR-AUC | Precision @ 80% recall | Brier | Threshold | Recall | Precision | F1 | Accuracy | Flagged |
|---|---|---|---|---|---|---|---|---|---|---|
| Cross-validation (out-of-fold) | 0.8244 | 0.4546 | 0.3300 | 0.1062 | 0.1508 | 80.0% | 33.0% | 0.467 | 71.3% | 38.2% |
| **Test set** | **0.8247** | 0.4633 | 0.3291 | 0.1057 | 0.1508 | **79.6%** | **33.0%** | 0.467 | 71.3% | 38.0% |

**Finding:** identical to CatBoost in `Modeling.ipynb` (step 1) to 4 decimals, including the threshold, and the
split matches `data/splits/train_test_split.csv` row for row. The script reproduces the notebook result.

## Step 5 — Risk levels and API (`ml/risk.py`, `api/main.py`, 2026-11-18)

### Calibration of the predicted probabilities (CatBoost, 5-fold out-of-fold, training rows)

| Predicted probability | People | Mean predicted | Actually have prediabetes/diabetes |
|---|---|---|---|
| 0–5% | 74,922 | 1.9% | 2.0% |
| 5–10% | 29,605 | 7.3% | 7.5% |
| 10–15.1% | 20,953 | 12.4% | 12.8% |
| 15.1–20% | 15,667 | 17.5% | 18.0% |
| 20–30% | 23,752 | 24.7% | 24.6% |
| 30–40% | 16,257 | 34.6% | 35.3% |
| 40–50% | 10,611 | 44.7% | 44.6% |
| 50–60% | 6,303 | 54.5% | 52.8% |
| 60–100% | 4,874 | 68.2% | 64.0% |

**Finding:** the probabilities are honest up to ~60% (slightly too high above that). A shown "30%" really means
about 30 of 100 similar people have prediabetes or diabetes; this confirms D-19. Highest predicted probability: 97.6%.

### Risk-level options (same out-of-fold predictions; overall 15.8% positive)

Low (below 15.1%): 61.8% of people, 5.1% of them positive, 20.0% of all positives.

| "High" starts at | Elevated: people | Elevated: positive | High: people | High: positive | High: share of all positives |
|---|---|---|---|---|---|
| 25% | 14.1% | 20.0% | 24.1% | 40.6% | 62.1% |
| **30%** (chosen, D-48) | **19.5%** | **22.0%** | **18.7%** | **44.5%** | **52.9%** |
| 35% | 23.9% | 24.1% | 14.3% | 47.9% | 43.4% |
| 40% | 27.5% | 25.9% | 10.7% | 51.3% | 34.9% |
| 45% | 30.3% | 27.5% | 7.9% | 54.2% | 27.1% |
| 50% | 32.7% | 28.8% | 5.5% | 57.7% | 20.2% |

### API

`python -m pytest tests`: **48 tests passed** (validation, training, risk levels, API); flake8 clean.

Manual check against the running server (`python -m api.main`): `/health` returns the loaded model
(`catboost_20261118_084406`). A person with high blood pressure, high cholesterol, BMI 31.4, general health 3 and
age group 9 (60–64) got **33% → high**, with the doctor message and the disclaimer. Invalid input (`HighBP` 2, `BMI`
as text, missing columns) got **HTTP 400** with all problems listed.

## Step 6 — Web UI (`ui/app.py`, 2026-11-18)

`python -m pytest tests`: **60 tests passed** (validation, training, risk levels, API, UI); flake8 clean. The UI tests
check that the form asks for every feature exactly once, that the answer codes match the allowed ranges, the BMI
calculation, that a complete form gives a payload the API accepts, that unanswered questions are listed, the
plain-language BMI error, and the marker position on the risk bar.

Checked with the API and UI running (`python -m api.main`, `python -m ui.app`):
- Screenshots (headless Chrome) of the page at 1280 px and 700 px wide, and of the result panel for real API answers:
  <1% → low, 14% → low, 33% → high, 75% → high, plus the "unanswered questions" message.
- The "Check my risk" logic against the live API: complete answers (175 cm, 96 kg → BMI 31.3) gave **33% → high**
  with the doctor message; a missing age and weight were listed; 250 cm / 30 kg (BMI 4.8) was rejected with a
  plain-language message.
- Fixed after the screenshots (D-57): "0%" → "less than 1%", threshold shown as 15.1%; added the viewport tag for
  phones. Narrow screenshots below ~550 px can't be taken with headless Chrome (it lays the page out at 548 px for a
  520 px window), so the phone layout should be checked once in a real browser (Chrome: F12 → device toolbar).

## Step 7 — Local deployment (`run.py`, `start.bat`, 2026-11-18)

`python -m pytest tests`: **61 tests passed** (incl. `tests/test_run.py`, which really starts and stops both servers
with `run.py --check`); flake8 clean.

- `start.bat --check` (run from PowerShell by its full path, as a double-click does): found the `da-ml` environment,
  started the API and the UI, both answered, both stopped; exit code 0.
- Starting a second time while the API was already running: "Port already in use: 127.0.0.1:5001. Is the app
  already running (another window)? Close it first." and exit code 1, without starting anything.

## Step 8 — Automatic deployment of new models (2026-11-18)

`python -m pytest tests`: **72 tests passed** (11 new: the quality check in 6 cases; good model → live, worse
model → saved but not live, `--force` → live; same-second runs; the API switching models without a restart; the API
keeping the old model when the new one is broken; the UI fetching the cut-offs again for a new model); flake8
clean.

Live check with the app running (`python run.py --no-browser`):

| | Result |
|---|---|
| Before | `/health` → `catboost_20261118_084406` |
| `python -m ml.train` (Kaggle CSV, ~2 min) | test ROC-AUC 0.8247, recall 79.6%, precision 33.0%; quality check passed (same scores as the live model) → "Now live: catboost_20261118_095453", exit code 0 |
| After, **no restart** | `/health`, `/model` and `/predict` all answered with `catboost_20261118_095453` (threshold 0.1508; the example person 33.2% → high); API log: "Switched to model catboost_20261118_095453"; UI still answered |

Afterwards `models/latest.txt` was set back to `catboost_20261118_084406` and the demo model folder deleted, so the
repository is unchanged.

### Live model label in the UI (D-65)

`python -m pytest tests`: **73 tests passed**; flake8 clean. With the app running, the header showed "Model
catboost_20261118_100430 · trained 18 Nov 2026, 10:04 · test ROC-AUC 0.825" (the model trained by the team with the
PyCharm "Train model" configuration). After `latest.txt` was pointed at `catboost_20261118_084406`, the page showed
that model ("trained 18 Nov 2026, 08:44"); `latest.txt` was then set back to `catboost_20261118_100430`.

## Step 9 — Data store and logging (`ml/store.py`, 2026-11-18)

`python -m pytest tests`: **82 tests passed** (9 new: tables created; predictions stored with no identifying columns
and UTC time; training runs imported once; rejected runs marked "not deployed"; unknown table names refused; the
API stores only with `?store=true`; a broken store doesn't stop predictions; all training runs recorded; the UI only
asks for storage when the checkbox is ticked); flake8 clean. The tests never created the real `data/store.db`.

Live check with the app running on a **temporary** store (`DA_ML_STORE`, so no test data entered the real one):
- Checkbox unticked → nothing stored; ticked → 1 row (`2026-11-18T09:21:55+00:00`, `catboost_20261118_100430`,
  0.3323, high, source `ui`, Age 9, BMI 31, …) and the result said "your answers were stored anonymously".
- Screenshot: the unticked checkbox sits above "Check my risk" in the result panel.

Real store `data/store.db` created with the two existing training runs (`catboost_20261118_084406`,
`catboost_20261118_100430`: test ROC-AUC 0.8247, recall 79.6%, both deployed) and 0 predictions.

## Step 10 — Monitoring (`ml/monitoring.py`, 2026-11-18)

`python -m pytest tests`: **92 tests passed** (10 new: PSI and its levels; bins cover every allowed value without
overlap; reference file; "not enough data" below 100; same population → stable; different user group → drift with
Age and BMI on top; model without reference; reference refused for the wrong data file; volume and error counting;
the API counts every request without answers); flake8 clean. The real store was unchanged after the tests.

`reference.json` built for both existing models. Reference of `catboost_20261118_100430`: 202,944 training rows,
50,736 test rows, mean predicted risk 15.7%, share flagged 38.0%.

`python -m ml.monitoring` on the real store: 0 requests, 0 stored predictions → **not enough data** (correct; no
one has used the app with storage yet).

Simulation on a temporary store (300 stored predictions, 312 requests incl. 12 invalid):

| | Same population (300 unseen survey people) | Older, heavier users (same people, age 70–80+, BMI 35–45) |
|---|---|---|
| Status | **stable** | **drift** |
| Prediction PSI | 0.030 | 1.674 |
| Mean risk (reference 15.7%) | 16.0% | 30.4% |
| Share flagged (reference 38.0%) | 38.7% | 75.3% |
| Largest question PSIs | BMI 0.070, Age 0.028, MentHlth 0.017 (all stable) | BMI 7.17, Age 6.44 (drift); all others < 0.03 |
| Error rate | 3.8% | 3.8% |

**Finding:** monitoring separates a normal user group from a shifted one, points at the questions that changed, and
shows the effect on the predictions (twice as many people flagged).

## Step 11 — Real-time dashboard (`ui/dashboard.py`, `/monitoring`, 2026-11-18)

`python -m pytest tests`: **100 tests passed** (8 new: `/monitoring` on an empty store and with data, aggregates
only; the page renders "not enough data", "drift", "stable" and "no reference"; the model history marks the live
model and shows why a model was rejected; "no connection" when the API is down; routing `/` and `/dashboard`);
flake8 clean. Real store unchanged after the tests.

Colour check (dataviz validator, light, white surface): blue `#2a78d6` + orange `#eb6834` → all checks pass (CVD
ΔE 24.7, normal-vision ΔE 33.6, contrast ≥ 3:1); blue + red `#d03b3b` → all pass; blue + grey `#b4b2a9` → **fail**
(chroma 0.013, contrast 2.13:1), so grey was not used.

Visual check on a **temporary demo store** (743 requests over 14 days, 5% invalid, 43% opted in):
- 400 stored predictions, a third of the users 2–4 age groups older → **Watch** (Age PSI 0.112, predictions stable,
  PSI 0.037, flagged 38.2% vs. 38.0%): the early warning before the predictions change.
- plus 250 users aged 75–79 with BMI ≥ 34 → **Drift** (Age 0.993, BMI 0.590; prediction PSI 0.156 "watch"; mean risk
  20.3% vs. 15.7%; flagged 52.9% vs. 38.0%).
- Screenshots at 1280 px and 760 px. Fixed after looking at them: table font, line breaks in the model history,
  legend order, gaps between stacked segments, white dashes on empty days, overlapping threshold labels, tile
  spacing.

## Step 12 — Closing the loop (2026-11-18)

`python -m pytest tests`: **116 tests passed** (16 new: labelled rows in training format; old stores get the new
columns; the API stores the diagnosis only with `store=true` for one person and rejects invalid values; the UI sends
the diagnosis only with consent and notes a known diagnosis; recommendation levels; `--check` exit code 3; the
dashboard shows the recommendation; training with user data combines and records the rows); flake8 clean. The real
store and `data/training_sets/` were untouched by the tests.

End-to-end run in **temporary folders** (copy of the live model, demo store): 400 unseen survey people aged 70+,
each with their true survey answer as diagnosis, used the app.

| Stage | Result |
|---|---|
| 1. Monitoring | **Drift**: Age PSI 6.404 (drift); Income 0.209, HighBP 0.160, MentHlth 0.129, AnyHealthcare 0.103 (watch); predictions PSI 1.050, mean risk 23.1% vs. 15.7%, flagged 61.5% vs. 38.0% → **Retraining recommended**, "400 stored user rows have a diagnosis …", `python -m ml.train --with-user-data` |
| 2. Retraining | 253,680 Kaggle rows + 400 user rows = 254,080 (combined CSV saved); threshold 0.1505; test ROC-AUC 0.8257, recall 80.0%, precision 33.1% → quality check passed → **live** (`catboost_20261118_110828`), exit code 0 |
| 3. Afterwards | Monitoring for the new model: 0 stored predictions → "not enough data"; model history: `catboost_20261118_100430` 0 user rows, `catboost_20261118_110828` 400 user rows |

**Findings:** the loop works end to end. With 400 user rows the new model is practically the old one (the reference is
still 99.8% survey data), so if users really are much older, drift returns once the new model has 100+ stored
predictions (D-80). Demo caveat: the 400 "users" were Kaggle test rows, so they were also in the Kaggle part of the
combined data; real users would be new people.

## Step 13 — Supplementary analyses (`ml/notebooks/Supplementary.ipynb`, 2026-11-23)

Checks from the course method for the extended abstract (D-82). Sections 1–2 below are final (they train nothing);
the feature selection and tuning results (sections 3–4) follow when the full run has been done.

### Confusion matrix of the live model (test set, 50,736 people)

`catboost_20261118_100430`, saved threshold 0.1508 (80% recall on out-of-fold predictions, D-23); test ROC-AUC 0.8247.

| | Flagged (see a doctor) | Not flagged | Total |
|---|---|---|---|
| **Has prediabetes/diabetes** | 6,365 | 1,630 | 7,995 |
| **Does not** | 12,920 | 29,821 | 42,741 |
| **Total** | 19,285 | 31,451 | 50,736 |

Recall 79.6%, precision 33.0%, specificity 69.8%, flagged 38.0%: **about 2 false alarms per real case found**. These are
the exact counts for the live model; the "caught / missed / false alarms" table in step 2 was derived from rounded
percentages of the CatBoost retrained in `AutoML.ipynb`.

### Outliers: Z-score and IQR (training rows)

Only the three numeric features can have outliers; the 18 binary/coded features are range-checked by the validation.

| Feature | Median | Q3 | Max | Skewness | \|z\| > 3 | IQR k = 1.5 | IQR k = 3 | Share of zeros |
|---|---|---|---|---|---|---|---|---|
| BMI | 27 | 31 | 98 | 2.11 | 1.2% (> 48.2) | 3.9% (> 41.5) | 0.7% (> 52) | 0% |
| MentHlth | 0 | 2 | 30 | 2.73 | 5.0% | 14.2% | 12.4% | 69.3% |
| PhysHlth | 0 | 3 | 30 | 2.21 | 0.0% | 16.1% | 13.3% | 63.2% |

**MentHlth / PhysHlth:** the IQR method flags 14–16% because the median is 0 (the course's warning about variables
with many zeros). The flagged values are a real group: diabetes rate by physically bad days 12.0% (0 days), 17.6%
(1–14), 29.7% (15–29), 32.5% (30); mentally bad days 15.0%, 14.7%, 21.9%, 26.1%. **Kept.**

**BMI** (all 253,680 rows):

| BMI | People | Diabetes rate | General health (1 good – 5 poor) | Difficulty walking |
|---|---|---|---|---|
| 25–29 | 93,749 | 13.0% | 2.41 | 13% |
| 30–34 | 53,451 | 21.7% | 2.69 | 20% |
| 40–44 | 8,099 | 35.3% | 3.16 | 37% |
| 50–54 | 1,190 | 42.6% | 3.44 | 56% |
| 60–64 | 199 | 44.2% | 3.79 | 71% |
| 65–69 | 70 | 51.4% | 3.89 | 74% |
| **70–98** | **599** | **15.5%** | **2.56** | **22%** |

The IQR outliers up to BMI 70 are real people with higher risk (diabetes rate 37–51%). The 599 rows with BMI 70–98 are
not: their diabetes rate and health profile match BMI 25–34, their number jumps up after falling steadily, and the
values are almost only odd numbers (71, 73, 75, …). Full table: `bmi_bands_actual_data.csv`.

**The live model has learned the error** (partial dependence: the same 5,000 test people, only BMI changed):

| BMI | 25 | 30 | 40 | 50 | 60 | 65 | 70 | 75 | 80 | 90 |
|---|---|---|---|---|---|---|---|---|---|---|
| Mean predicted risk | 11.6% | 16.5% | 25.7% | 28.8% | 25.8% | 33.3% | 16.6% | 13.4% | 13.9% | 12.5% |

A user entering BMI 75 gets about the risk of BMI 30. The curve is noisy between BMI 55 and 70 (few people per value,
e.g. 19 at BMI 65). → **Remove BMI ≥ 70 from training** (D-83). Figure: `bmi_outlier_fund.png/.pdf` (Danish, for the
report).

### Feature engineering (5-fold CV, training rows without BMI ≥ 70: 202,464 rows, 15.76% positive)

Engineered: `n_conditions`, `bad_days`, `lifestyle_score`, HighBP × HighChol, BMI × Age, GenHlth × Age (D-84).

| Model | Features | ROC-AUC | PR-AUC | Precision @ 80% recall | Brier | CV time |
|---|---|---|---|---|---|---|
| Logistic regression | 21 | 0.8207 | 0.4484 | 0.3263 | 0.1070 | 5 s |
| Logistic regression | 21 + 6 engineered | 0.8218 | 0.4500 | 0.3270 | 0.1068 | 15 s |
| **CatBoost** | **21** | **0.8240** | **0.4550** | **0.3296** | **0.1063** | 56 s |
| CatBoost | 21 + 6 engineered | 0.8240 | 0.4536 | 0.3291 | 0.1063 | 49 s |

**Finding:** no gain for CatBoost (ROC-AUC ±0.0000, PR-AUC −0.0014); +0.0010 for logistic regression, which cannot
learn interactions by itself. → **No engineered features** (D-84). Side note: logistic regression scored 0.8207 here
vs. 0.8192 on all rows in step 1, possibly because the BMI ≥ 70 rows distort its single BMI weight; different rows
and folds, so to be checked in the final run (D-85).

## Expected vs. actual

Predictions from `Diabetes_Prediction_Feasibility.md` (written before any model was trained) compared with the
measured test-set results:

| Prediction | Actual |
|---|---|
| ROC-AUC ~0.80–0.83 | 0.819 (logistic regression) – 0.826 (AutoGluon) ✅ |
| At 75–80% recall, ~30–35% precision | 79.6% recall at 33.0% precision ✅ |
| Accuracy then ~70–75% | 71.3% ✅ |
| Models within ~0.01–0.02 ROC-AUC of each other | Strong models within 0.002; logistic regression 0.006 behind ✅ |
| Prediabetes alone too rare to predict | Merged into the yes/no target before training (D-08) |
