# How the Model Works

How the deployed model (`models/catboost_20261118_084406/`) turns a person's survey answers into a risk percentage
and a message, why the approach is sound, and where its limits are.

Related documents: [`PROJECT_PLAN.md`](PROJECT_PLAN.md) (plan) · [`DECISIONS.md`](DECISIONS.md) (what was decided
and why) · [`RESULTS.md`](RESULTS.md) (all measurements) ·
[`Diabetes_Prediction_Feasibility.md`](Diabetes_Prediction_Feasibility.md) (expectations before training, and an
introduction to CatBoost).

## 1. In one sentence

The model adds up learned values from 500 decision trees for the person's combination of answers, turns the sum into
a probability, and that probability is compared with two cut-offs to decide whether the person is flagged and which
message they see.

```
answers ──► 500 trees: each picks one leaf value ──► sum (raw score) ──► sigmoid ──► probability, e.g. 33.2%
                                                                                         │
                                         compare with 15.1% ──► flagged yes/no (screening decision, D-23)
                                         compare with 15.1% and 30% ──► low / elevated / high (message, D-48)
```

## 2. Where the percentage comes from

### The trees

The model (CatBoost, D-22) consists of **500 decision trees**, each **6 questions deep**. Every tree asks yes/no
questions about the answers (for example "HighBP = 1?", "Age ≥ 8?", "BMI > 30?") and ends in a **leaf** that holds
a small number, the **leaf value**: positive means more risk, negative means less. A tree's leaf depends on a
*combination* of answers, so the model can capture effects like "high BMI **and** high blood pressure together".

The leaf values were **learned during training** (`ml/train.py`) from the 202,944 training rows: CatBoost built the
trees one at a time, each new tree correcting the errors of the trees before it. The training goal (*log loss*)
rewards **honest probabilities**, not just right/wrong answers. After training, the leaf values are fixed and saved
in `model.cbm`; a prediction only looks them up and adds them, which takes milliseconds.

### From the sum to a percentage

For one person, the 500 selected leaf values are added up:

```
raw score = tree 1 + tree 2 + … + tree 500
```

The sum is on a **log-odds** scale (convenient for adding, not readable). A fixed formula, the **sigmoid**, turns it
into a probability between 0% and 100%:

```
probability = 1 / (1 + e^(−raw score))
```

### What the percentage means

**"33%" means: of 100 people with answers like these, about 33 have prediabetes or diabetes.** This was checked on
people the model had not seen: of everyone it gave 30–40%, 35.3% actually had prediabetes or diabetes (calibration
table in `RESULTS.md`, step 5).

## 3. Worked example

The example person from the API check in step 5. Answers: high blood pressure yes, high cholesterol yes, cholesterol
checked yes, BMI 31.4 (rounded to 31, D-38), smoker no, stroke no, heart disease no, physical activity yes, fruit
yes, vegetables yes, heavy drinking no, health coverage yes, couldn't afford a doctor no, general health 3 ("good"),
2 days of poor mental health, 5 days of poor physical health, difficulty walking no, male, age group 9 (60–64),
education 5 (some college), income 6 ($35–50k).

### The raw score and the percentage

```
raw score = −0.698   →   1 / (1 + e^0.698) = 0.3323 = 33.2%   →   ≥ 30%  →  "high", flagged
```

How the estimate settles as trees are added (the early trees do the rough work, later ones fine-tune):

| Trees used | Raw score | Probability |
|---|---|---|
| 1 | −0.027 | 49.3% |
| 10 | −0.547 | 36.7% |
| 50 | −0.646 | 34.4% |
| 100 | −0.605 | 35.3% |
| 250 | −0.670 | 33.8% |
| 500 (all) | −0.698 | **33.2%** |

### Which answers raised or lowered the risk

The same sum can be split up **per answer** (SHAP values, computed with CatBoost's `get_feature_importance(...,
type="ShapValues")`). It starts from the score of a "typical" person and adds one push per answer:

| Answer | Push (log-odds) | Running risk |
|---|---|---|
| Starting point (typical person) | −2.436 | 8.0% |
| High blood pressure = yes | +0.396 | 11.5% |
| General health = 3 ("good") | +0.379 | 16.0% |
| High cholesterol = yes | +0.355 | 21.3% |
| Age group 9 (60–64) | +0.300 | 26.8% |
| BMI 31 | +0.264 | 32.3% |
| Sex = male | +0.072 | 33.8% |
| Cholesterol checked = yes | +0.046 | 34.9% |
| Mental health: 2 bad days | −0.042 | 33.9% |
| The other 13 answers combined | −0.032 | **33.2%** |

The pushes add up exactly to the raw score (−2.436 + 1.738 = −0.698). The pushes themselves are fixed; the "running
risk" column depends on the order in which they are added and is only there for illustration.

**Why is the "typical person" 8% and not the average of 15.8%?** Most people have a low risk and a small group a
very high one. The typical person (on the log-odds scale) is therefore lower than the plain average, which is pulled
up by the high-risk group.

## 4. The probability is the model's output; yes/no is derived

The model only produces the **probability**. It knows nothing about the 15.1% threshold or the 30% cut-off; those
were chosen afterwards:

- **15.1% (screening threshold, D-23):** chosen so that 80% of people with prediabetes/diabetes are above it. It is
  re-chosen on every training run and stored in `metadata.json`. With a different recall target, the same model
  would give different yes/no answers.
- **30% ("high" cut-off, D-48):** only splits the flagged people into "elevated" and "high" to change the wording of
  the message. It does **not** change who is flagged, so it doesn't change false positives or false negatives.

The metrics follow the same split (D-26):

| Judges the probability itself | Judges the yes/no after the threshold |
|---|---|
| ROC-AUC, PR-AUC (ranking), Brier score (honesty) | Recall, precision, F1, accuracy, share flagged |

That's why models were compared on the left-hand metrics first and the threshold was chosen afterwards.

## 5. Why this is the right approach

**Model and decision are separate.** The model estimates risk as honestly as possible; the threshold is a *policy*
("we want to catch 80%"). So:
- the policy can be changed (e.g. 85% recall) **without retraining**,
- different thresholds can be used for different purposes (e.g. a campaign with a limited number of follow-ups),
- the yes/no rests on a meaningful, checked number instead of a black box.

A model that only output yes/no would hide this trade-off.

**It matches established screening tools.** Questionnaire-based diabetes risk tests, such as the American Diabetes
Association's risk test or the Finnish FINDRISC score, work the same way: answers → risk score → category → "talk
to your doctor" above a certain level. The difference is that they add up hand-made points, while this model
learned its "points" (the leaf values) from about 200,000 people.

**The standard safeguards are in place:**
- the probabilities were checked for calibration (`RESULTS.md`, step 5),
- the threshold was chosen on out-of-fold predictions, not on the test set (D-23),
- the test set was used once, and its scores matched cross-validation (D-24),
- accuracy was not used as the target, since it is misleading with 84% "no" (D-13),
- the probabilities were kept honest by not balancing the training data (D-19).

## 6. Limits

These are limits of the data and the scope, not flaws in the method:

- **One population.** The data is US adults in 2015, self-reported in a phone survey (BRFSS). The probabilities are
  calibrated for *that* population. Where diabetes is less common (another country, a younger user group), the
  percentages will probably be **too high**, and the 80% recall won't be exactly 80%. Fixing that would need data
  from the target population.
- **No lab values.** The model estimates risk for *people with similar survey answers*; it doesn't measure anything
  about the person's body. It cannot diagnose (D-07).
- **Snapshot, not a forecast.** The survey shows who has diabetes now, not who will develop it. Some answers
  (heart disease, difficulty walking, poor general health) may be *results* of diabetes rather than causes.
- **Slight overestimation at the top.** Above ~60%, predictions are a little too high (68% predicted vs. 64% actual).
  Minor; could be corrected with an extra calibration step if it ever matters.
- **A low result doesn't rule anything out.** 20% of people with prediabetes/diabetes are in the "low" group, which
  is why the low message still mentions a doctor (D-49).
- **One test split, no uncertainty ranges.** Scores such as ROC-AUC 0.825 come from a single 80/20 split. Ranges
  (e.g. "0.825 ± 0.004") could be added with bootstrapping if the results need formal reporting.
- **Regulation.** If the tool were offered to the public beyond a study project, software that gives personal
  health-risk advice may count as a **medical device** in the EU, with formal requirements. For a project and demo
  this doesn't apply.
