# Results

All **measured** findings of the project, step by step. For *what was decided because of them* (and why), see
[`DECISIONS.md`](DECISIONS.md); for the plan, see [`PROJECT_PLAN.md`](PROJECT_PLAN.md); for the expectations
before any model was trained, see [`Diabetes_Prediction_Feasibility.md`](Diabetes_Prediction_Feasibility.md).

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
