import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# Then later:
#from tensorflow.keras.models import load_model

st.set_page_config(page_title="🌌 NASA Exoplanet Classifier", layout="wide")

try:
    model = tf.keras.models.load_model("my_model2.h5")
    st.sidebar.success("✅ Model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"❌ Failed to load model: {e}")
    model = None

st.title("🌌 NASA Exoplanet Classifier")
st.markdown("""
Welcome to the AI-powered Exoplanet Classifier! 🚀  
This tool uses NASA's open exoplanet datasets (Kepler, K2, TESS)  
to automatically identify exoplanet candidates.

It can classify each observation into:
- 🪐 Confirmed Exoplanet
- 🧩 Candidate
- ❌ False Positive

You can upload a CSV file or enter values manually below.
""")

st.sidebar.header("⚙️ Input Settings")
input_mode = st.sidebar.radio("Choose Input Type:", ["Manual Entry", "Upload CSV"])

features = [
    'pl_orbper', 'pl_tranmid', 'pl_trandep', 'pl_trandur',
    'pl_rade', 'pl_eqt', 'pl_insol', 'st_teff', 'st_rad',
    'st_logg', 'pl_ratror', 'st_mass'
]

default_values = {
    'pl_orbper': 365.25,
    'pl_tranmid': 2454833.0,
    'pl_trandep': 0.008,
    'pl_trandur': 13.0,
    'pl_rade': 1.0,
    'pl_eqt': 288.0,
    'pl_insol': 1.0,
    'st_teff': 5800.0,
    'st_rad': 1.0,
    'st_logg': 4.4,
    'pl_ratror': 0.1,
    'st_mass': 1.0
}

if input_mode == "Manual Entry":
    st.subheader("🧩 Enter Feature Values")
    user_input = {}
    cols = st.columns(2)
    for i, f in enumerate(features):
        with cols[i % 2]:
            user_input[f] = st.number_input(f"{f}", value=float(default_values[f]), format="%.5f")
    df = pd.DataFrame([user_input])
    if st.button("🔮 Predict"):
        if model is not None:
            preds = model.predict(df)
            confidence = np.max(preds)
            pred_class = np.argmax(preds)
            classes = ["❌ False Positive", "🧩 Candidate", "✅ Confirmed Exoplanet"]
            label = classes[pred_class] if pred_class < len(classes) else f"Class {pred_class}"
            st.success(f"**Prediction:** {label}")
            st.write(f"**Confidence:** {confidence * 100:.2f}%")
        else:
            st.error("Model not loaded correctly. Please check your model file.")
else:
    st.subheader("📁 Upload CSV File")
    uploaded = st.file_uploader("Upload your CSV file with feature columns", type=["csv"])
    if uploaded is not None:
        data = pd.read_csv(uploaded)
        st.write("📄 Preview of uploaded data:")
        st.dataframe(data.head())
        if 'disposition' in data.columns:
            data = data.drop(columns=['disposition'])
        data = data[features]
        if model is not None:
            preds = model.predict(data)
            confidences = np.max(preds, axis=1)
            predicted_classes = np.argmax(preds, axis=1)
            classes = ["False Positive", "Candidate", "Confirmed Exoplanet"]
            data['Prediction'] = [classes[i] if i < len(classes) else f"Class {i}" for i in predicted_classes]
            data['Confidence'] = confidences
            st.success("✅ Predictions completed successfully!")
            st.dataframe(data[['Prediction', 'Confidence']])
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results", csv, "exoplanet_predictions.csv", "text/csv", key='download-csv')
        else:
            st.error("Model not loaded correctly. Please check your model file.")

st.markdown("---")
st.markdown("👨‍🚀 Built by AstroAgents Team | Powered by NASA Data 🌠")