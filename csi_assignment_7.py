# app.py

import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

# —— Load model relative to script location ——
MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"

@st.cache(allow_output_mutation=True)
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

model = load_model()

# —— UI ——
st.title("🌼 Iris Flower Species Predictor")
st.write("Use the sliders in the sidebar to input features and predict the species.")

# Sidebar sliders
sl = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sw = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
pl = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
pw = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

input_df = pd.DataFrame({
    "sepal_length": [sl],
    "sepal_width":  [sw],
    "petal_length": [pl],
    "petal_width":  [pw]
})

st.subheader("🌸 Your Input Features")
st.write(input_df)

# —— Prediction ——
if st.button("Predict"):
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    st.subheader("🎯 Prediction")
    st.write(f"**{pred}**")

    st.subheader("📊 Prediction Probability")
    proba_df = pd.DataFrame([proba], columns=model.classes_)
    st.write(proba_df)
