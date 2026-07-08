"""
Streamlit web app: Lower-Limb Amputation Risk Prediction in Diabetic Foot Ulcer Patients
==========================================================================================
Loads the trained CatBoost model + Platt (sigmoid) calibrator + decision threshold,
takes patient predictor values as input, and returns:
  - Binary prediction (Amputation risk: Yes/No)
  - Calibrated predicted probability

Run locally:
    streamlit run app.py
"""

import json
import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostClassifier
import joblib

st.set_page_config(page_title="ALARM-DFU", page_icon="🦶", layout="centered")

@st.cache_resource
def load_artifacts():
    with open("model_config.json") as f:
        config = json.load(f)
    model = CatBoostClassifier()
    model.load_model("catboost_model.cbm")
    platt = joblib.load("platt_calibrator.joblib")
    return config, model, platt


def apply_platt(platt, raw_probs):
    eps = 1e-6
    raw_clipped = np.clip(raw_probs, eps, 1 - eps)
    logit = np.log(raw_clipped / (1 - raw_clipped)).reshape(-1, 1)
    return platt.predict_proba(logit)[:, 1]


config, model, platt = load_artifacts()
CAT_FEATURES = config["categorical_features"]
NUM_FEATURES = config["numerical_features"]
ALL_FEATURES = config["all_features"]
THRESHOLD = config["decision_threshold"]
CATEGORY_LEVELS = config["category_levels"]
NUM_RANGES = config["numerical_ranges"]

st.title("Amputation Likelihood Assessment & Risk Model for Diabetic Foot Ulcers (ALARM-DFU)")
st.caption(
    "CatBoost model for predicting lower-limb amputation risk in hospitalized "
    "diabetic foot ulcer patients."
)

st.markdown("---")
st.subheader("Patient Information")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    inputs = {}

    with col1:
        st.markdown("**Categorical predictors**")
        for feat in CAT_FEATURES:
            options = CATEGORY_LEVELS[feat]
            inputs[feat] = st.selectbox(feat, options)

    with col2:
        st.markdown("**Numerical predictors**")
        for feat in NUM_FEATURES:
            rng = NUM_RANGES[feat]
            inputs[feat] = st.number_input(
                feat,
                min_value=0.0,
                value=float(rng["median"]),
                step=0.1,
                help=f"Observed range in training data: {rng['min']:.1f} – {rng['max']:.1f}",
            )

    submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

if submitted:
    X_new = pd.DataFrame([inputs])[ALL_FEATURES]
    for c in CAT_FEATURES:
        X_new[c] = X_new[c].astype(str)

    raw_prob = model.predict_proba(X_new)[:, 1]
    calibrated_prob = apply_platt(platt, raw_prob)[0]
    prediction = int(calibrated_prob >= THRESHOLD)

    st.markdown("---")
    st.subheader("Result")

    c1, c2 = st.columns(2)
    with c1:
        if prediction == 1:
            st.error("**Predicted: Amputation risk = YES**")
        else:
            st.success("**Predicted: Amputation risk = NO**")
    with c2:
        st.metric("Predicted probability", f"{calibrated_prob:.1%}")

    st.progress(min(max(calibrated_prob, 0.0), 1.0))
    st.caption(
        f"Decision threshold: {THRESHOLD:.3f}. Probability ≥ threshold → classified as 1 (Amputation)."
    )

    st.warning(
        "⚠️ This tool is for research purposes only and is **not validated for clinical decision-making**. "
        "It must not be used to guide individual patient care without further external validation."
    )
