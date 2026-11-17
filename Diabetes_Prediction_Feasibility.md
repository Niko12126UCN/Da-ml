# Can We Predict Diabetes from the BRFSS 2015 Health Indicators Dataset?

**Dataset:** [Kaggle — Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset/data)
(`diabetes_012_health_indicators_BRFSS2015.csv`)
**Based on:** the dataset itself, `EDA.ipynb` and the YData Profiling report `Diabetes_EDA_Profile.html`.
No model was trained for this assessment.

> **Update 2026-11-17:** models have since been trained. The predictions below held up; see the
> "Expected vs. actual" section in [`RESULTS.md`](RESULTS.md).

## Short answer

**Yes, it's feasible, but only for a specific job.** A model trained on this data can **estimate a person's risk**
of having diabetes or prediabetes from simple survey answers. It **cannot diagnose** diabetes.

## Why it's feasible

### 1. There's plenty of data

There are 253,680 people and 21 features, with no missing values and every coded column inside its allowed range.
Even the smallest group, diabetes (13.9%), has about 35,000 examples.

### 2. Several features clearly relate to diabetes, and the risk rises steadily with them

Share of people with prediabetes or diabetes at each level:

| Feature | Lowest level → highest level |
|---|---|
| `GenHlth` (1 = excellent … 5 = poor) | 3% → 41% |
| `BMI` (≤25 … >40) | 7% → 37% |
| `Age` (18–24 … 70–74) | 2% → 24% |
| `HighBP` (no / yes) | 7% → 27% |
| `HighChol` (no / yes) | 9% → 25% |
| `HeartDiseaseorAttack` (no / yes) | 14% → 36% |
| `Income` (lowest … highest) | 28% → 9% |

These patterns fit what medicine already knows about diabetes risk (obesity, high blood pressure, cholesterol,
age, inactivity, lower income). That suggests the data is real signal and not noise.

### 3. Each feature is only a weak signal on its own

The strongest single correlation with the target is about 0.30 (`GenHlth`), and most are below 0.2.
A model is useful precisely because it can combine many weak signals, such as an older person with high BMI,
high blood pressure and poor self-rated health. Published work on this exact Kaggle dataset typically reports
a ROC-AUC of about 0.80–0.83 (from general knowledge, not from anything run here).

## Limits on what it can do

- **It can't diagnose.** There are no lab values such as blood glucose or HbA1c, which are what actually define
  diabetes. All answers are self-reported in a phone survey, which adds noise.
- **It shows who has diabetes now, not who will get it.** The survey is a snapshot from 2015. A model can say
  "this person probably has diabetes", not "this person will develop it". Some features may even be *results*
  of diabetes: heart disease, difficulty walking and poor general health are common complications.
- **The prediabetes class is very hard to predict.** It's only 1.8% of the data (4,631 people). Many people with
  prediabetes don't know they have it, so some of the "0" answers are really prediabetes. Merging it with
  diabetes into a yes/no target (the dataset's `diabetes_binary` file) is the realistic choice.
- **The classes are imbalanced (84% no diabetes).** A model that always says "no diabetes" is already 84%
  accurate. Accuracy is therefore the wrong measure; use recall, precision, F1 and ROC-AUC, and consider class
  weights or the balanced 50/50 file.
- **There's a ceiling on accuracy.** 1,834 identical answer patterns appear with *different* diabetes labels, so
  some people simply can't be told apart from these features. Accuracy will level off at a point that no choice
  of model can push past.

## Bottom line

A **screening tool** is a realistic goal: something that flags people at higher risk so they can get a blood
test. The data is large and clean, the risk factors behave as medicine predicts, and combining them should give
useful but imperfect separation.

A sensible setup is:

- the yes/no target,
- a tree-based model such as CatBoost (already in `environment.yml`),
- evaluation by recall and ROC-AUC rather than accuracy.

Expect a good risk score, not a diagnosis.

---

## Expected performance

Expect roughly **85–87% accuracy on the full dataset**, but that figure is misleading, because a model that always
answers "no diabetes" already gets about 84% here. A better guide is **ROC-AUC of about 0.82**, or **about 75%
accuracy on the balanced 50/50 file**. These numbers come from published work and Kaggle notebooks on this dataset,
not from anything trained here.

Assuming the yes/no target (prediabetes and diabetes merged):

| Metric | Expected | Baseline (always "no diabetes") |
|---|---|---|
| Accuracy, full data (~84% "no") | ~85–87% | ~84% |
| Accuracy, balanced 50/50 file | ~74–76% | 50% |
| ROC-AUC | ~0.80–0.83 | 0.50 |

**Why accuracy is misleading here:** on the full data the model mostly predicts "no diabetes", because that's
usually right. It reaches high accuracy while missing most of the people who actually have diabetes. The 50/50 file
removes that shortcut, so its ~75% shows how well the model really tells the two groups apart.

### The trade-off that matters more

A decision threshold sets how high the model's risk score must be before it flags someone. A screening tool would
use a low threshold so it catches most people with diabetes. Roughly:

- If it catches **about 75–80% of diabetics** (recall),
- only **about 30–35% of the people it flags actually have diabetes** (precision).
- Overall accuracy then drops to about 70–75%.

For screening that's acceptable: a false alarm costs one blood test, and a missed case can go undiagnosed for
years. It's also why accuracy shouldn't be the target metric.

### Why it can't go much higher

- **Some people can't be told apart.** The 1,834 identical answer patterns with different labels set a hard ceiling.
- **The answers are coarse.** Most features are yes/no survey questions with no lab values.
- **The model choice matters little.** On this kind of data, logistic regression, random forests and CatBoost
  usually land within about 0.01–0.02 ROC-AUC of each other. Better features would help far more than a better model.

**Separating three classes** (none, prediabetes, diabetes) is much harder: prediabetes is detected almost never, so
the model is in practice still deciding diabetes vs. no diabetes.

---

## CatBoost explained

CatBoost is a machine-learning library from Yandex that builds a prediction out of **hundreds of small decision
trees**, where each new tree corrects the mistakes of the trees before it. The method is called **gradient
boosting**, and "Cat" stands for **categorical**: CatBoost is especially good with category-type features.

### Step 1: one decision tree

A decision tree is a chain of yes/no questions:

```
            HighBP = 1 ?
           /            \
         no              yes
         /                \
    BMI > 30 ?        GenHlth ≥ 4 ?
     /     \            /       \
  risk 5%  risk 15%  risk 20%  risk 40%
```

One tree is easy to understand but weak. It only asks a few questions, so its predictions are rough.

### Step 2: boosting, many trees learning from mistakes

Instead of one big tree, CatBoost builds trees **one after another**:

1. **Tree 1** makes a rough guess for everyone (e.g. "older + high BP = higher risk").
2. CatBoost checks **where it was wrong**: which diabetics got a low score, and which healthy people got a high one.
3. **Tree 2** is trained only to fix those errors.
4. **Tree 3** fixes what trees 1 and 2 still get wrong, and so on, often for 500–1000 trees.

The final prediction is the **sum of all the trees' small adjustments**, turned into a probability such as "this
person has a 34% risk".

It's like a team of weak students, where each new student studies only the questions the previous ones got wrong.
Together they score far better than any one of them. "Gradient" refers to the maths used to measure the errors and
decide which way each new tree should push the prediction.

### What makes CatBoost different

Other boosting libraries such as XGBoost and LightGBM work the same way. CatBoost adds three things:

- **It handles categories directly.** You can tell it a column is categorical (e.g. `Age` group, `Income` level,
  `GenHlth`), and it encodes the column internally in a way that avoids **target leakage**, i.e. the model quietly
  peeking at the answer while it learns. With other libraries you usually have to convert categories to numbers
  yourself first.
- **Ordered boosting.** Each tree's errors are measured on data that tree didn't learn from, which reduces
  **overfitting** (memorising the training data instead of learning general patterns).
- **Symmetric trees.** Every tree asks the same question at every split on a level. That makes it fast at
  prediction time and harder to overfit.

In practice, CatBoost often gives good results **with its default settings** and little tuning.

### Why it suits this dataset

| Dataset property | How CatBoost handles it |
|---|---|
| Mostly categorical / ordinal codes (`GenHlth`, `Age`, `Income`, `Education`) | Native categorical support |
| Risk depends on **combinations** (older + high BMI + high BP) | Trees naturally capture combinations of features |
| Non-linear patterns (risk rises with age, then dips at 80+) | Trees don't assume straight-line relationships |
| Class imbalance (84% "no") | `auto_class_weights="Balanced"` gives the rare class more weight |
| 250k rows | Fast enough on a normal laptop |
| You want to know *why* | Built-in feature importance and SHAP values show which features drove each prediction |

### Things to know

- **It's harder to interpret than logistic regression.** You can't read the model directly as a formula; you rely
  on feature importance and SHAP values instead.
- **It's not magic.** On survey data like this it will probably beat logistic regression by only a few hundredths
  of ROC-AUC. Its advantage is mostly convenience and robustness.
- **Main settings to tune:** `iterations` (number of trees), `learning_rate` (how big each correction step is),
  `depth` (questions per tree), plus early stopping so it stops adding trees once a validation set stops improving.
  `optuna`, already in `environment.yml`, can search these automatically.

**In short:** CatBoost builds many small trees in sequence, each fixing the previous ones' errors, and adds them up
into a risk score. It works with category-type features out of the box, which makes it a strong first choice for
this dataset.

---

## Methods to compare

CatBoost is the main candidate, but it should be compared against other methods. Each one below answers a
different question, from "is a simple model enough?" to "does a different kind of model find more signal?".

| # | Method | Type | Why include it | Package |
|---|---|---|---|---|
| 0 | **Dummy classifier** | Always predicts the most common class | The floor: any real model must beat it | `scikit-learn` ✅ |
| 1 | **Logistic regression** | Linear model | Simple, fast and fully interpretable (each feature gets a weight). If it's close to CatBoost, the simple model may be preferable | `scikit-learn` ✅ |
| 2 | **Decision tree** | Single tree | Shows how far one readable tree gets, and why combining many trees helps | `scikit-learn` ✅ |
| 3 | **Random forest** | Many trees built **in parallel** (bagging) | The classic alternative to boosting: trees vote instead of correcting each other | `scikit-learn` ✅ |
| 4 | **HistGradientBoosting** | Gradient boosting | scikit-learn's own fast boosting (LightGBM-style). Tests whether CatBoost's extras matter | `scikit-learn` ✅ |
| 5 | **CatBoost** | Gradient boosting with native categoricals | Main candidate (see above) | `catboost` ✅ |
| 6 | **XGBoost / LightGBM** *(optional)* | Gradient boosting | The two most popular boosting libraries, useful as a direct comparison | ❌ not in `environment.yml` yet |
| 7 | **Neural network (MLP)** *(optional)* | Neural network | Tests whether a very different model finds extra signal. Usually not better on tabular data like this | `scikit-learn` ✅ (`MLPClassifier`) |

✅ = already installed in the `da-ml` environment. XGBoost or LightGBM would need to be added to `environment.yml`.

### How to make the comparison fair

- **Same data split for every model:** one held-out test set (e.g. 20%, stratified so each split keeps the same share
  of diabetics). Touch it only once, at the very end.
- **Cross-validation on the training part** (e.g. 5-fold, stratified) to compare models and tune them, so one lucky
  split can't decide the winner.
- **Same target:** the yes/no target (prediabetes merged with diabetes).
- **Same handling of imbalance:** class weights (`class_weight="balanced"` / `auto_class_weights="Balanced"`) for
  every model that supports them.
- **Preprocessing where a model needs it:** logistic regression and the MLP need **scaled** numeric features
  (`StandardScaler`) and ideally one-hot encoded categories; tree models don't.
- **Similar tuning effort:** give each model a comparable `optuna` budget, or compare all on default settings first.
- **Decide about duplicates before splitting:** otherwise identical rows can end up in both train and test and make
  the scores look better than they are.

### What to measure

| Metric | Why |
|---|---|
| **ROC-AUC** | Main metric: how well the model ranks diabetics above non-diabetics, independent of threshold |
| **PR-AUC** (average precision) | Better than ROC-AUC at showing performance on the rare class |
| **Recall** at a chosen threshold | For screening: what share of diabetics are caught |
| **Precision** at the same threshold | What share of the flagged people actually have diabetes |
| **F1** | One number balancing recall and precision |
| **Accuracy** | Report it, but compare it against the dummy baseline (~84%) |
| **Training time** | Practical cost |

### What to expect from the comparison

Based on published results for this dataset, the models will probably land close together: the boosting models
(CatBoost, HistGradientBoosting, XGBoost/LightGBM) at the top around 0.82–0.83 ROC-AUC, logistic regression and
random forest slightly behind, and the single decision tree clearly lower. If logistic regression is within about
0.01 of CatBoost, it may be the better practical choice because it's simpler to explain. That is a useful outcome
in itself.
