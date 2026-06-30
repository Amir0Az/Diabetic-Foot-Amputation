# Lower-Limb Amputation Risk Prediction — Diabetic Foot Ulcer Patients

A CatBoost classifier predicting lower-limb amputation risk, with sigmoid (Platt)
probability calibration and an accuracy-optimized decision threshold, deployed as a
Streamlit web app.

## Files

| File                       | Purpose                                                         |
|-----------------------------|------------------------------------------------------------------|
| `data.csv`                  | Training data (221 patients, 10 predictors + outcome)            |
| `train_model.py`            | Trains the model, runs 5-fold CV, saves all artifacts            |
| `app.py`                    | Streamlit web app for making predictions                         |
| `catboost_model.cbm`        | Saved final CatBoost model (trained on all 221 instances)        |
| `platt_calibrator.joblib`   | Saved sigmoid (Platt) calibrator                                 |
| `model_config.json`         | Feature lists, category levels, threshold, CV metrics            |
| `requirements.txt`          | Python dependencies                                              |

## Methodology summary (for your Methods section)

- **Model**: CatBoostClassifier, 50 trees, learning rate 0.1, depth 5, Bernoulli
  bootstrap with subsample = 0.5, L2 leaf regularization (lambda) = 3. The six
  categorical predictors are passed natively to CatBoost (no one-hot encoding).
- **Calibration**: Sigmoid (Platt) scaling — a logistic regression fit on the
  logit of out-of-fold CatBoost probabilities versus the true outcome.
- **Cross-validation**: 5-fold stratified CV was used to obtain honest,
  out-of-sample performance metrics. For each of the 5 outer folds, a CatBoost
  model was trained on the other 4 folds; within that training portion, a nested
  5-fold CV generated out-of-fold raw probabilities used to fit the Platt
  calibrator for that fold. The calibrated, held-out predictions from all 5 outer
  folds were pooled to compute the metrics below and to select the decision
  threshold (the threshold that maximizes accuracy on these pooled out-of-fold
  predictions).
- **Final deployed model**: trained on **all 221 instances** (no data held out),
  using the same nested-CV calibration procedure on the full dataset to fit the
  Platt calibrator. The decision threshold from the CV step above is carried
  over to this final model.

### 5-fold cross-validated performance

| Metric | Value |
|---|---|
| AUC | 0.876 |
| Accuracy | 0.833 |
| Sensitivity | 0.775 |
| Specificity | 0.871 |
| PPV | 0.802 |
| NPV | 0.852 |
| F1 | 0.789 |
| Brier score | 0.136 |
| **Decision threshold** | **0.491** |

Confusion matrix (pooled out-of-fold): TN=115, FP=17, FN=20, TP=69.

These numbers are also stored in `model_config.json` under `cv_metrics`, and are
reprinted every time you re-run `train_model.py`.

## Re-running training

```bash
pip install -r requirements.txt
python train_model.py
```

This regenerates `catboost_model.cbm`, `platt_calibrator.joblib`, and
`model_config.json`. Because all random seeds are fixed (seed = 42), results are
reproducible.

## Running the web app locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

This opens the app at `http://localhost:8501`. Fill in the patient's predictor
values and click **Predict** to get the binary classification (0/1) and the
calibrated probability of amputation.

## Free online deployment (Streamlit Community Cloud)

1. Create a free account at https://share.streamlit.io (sign in with GitHub).
2. Create a new **public** GitHub repository and push these files to it:
   `app.py`, `requirements.txt`, `catboost_model.cbm`, `platt_calibrator.joblib`,
   `model_config.json`. (`train_model.py` and `data.csv` are optional to include,
   but harmless if you do.)
3. In Streamlit Community Cloud, click **New app**, select the repository,
   branch, and set the main file path to `app.py`.
4. Click **Deploy**. The app will build automatically using `requirements.txt`
   and give you a public URL (e.g. `https://your-app-name.streamlit.app`) that
   anyone can use to make predictions — no installation needed on their end.

Alternative free hosts with the same workflow: Hugging Face Spaces (choose the
"Streamlit" SDK when creating a Space) or Render's free web service tier.

## Important caveat

This model was trained and internally cross-validated on a single-center
dataset of 221 patients. It has **not** been externally validated. Before any
clinical use, it requires external validation on an independent cohort. The app
itself displays this warning to end users.
