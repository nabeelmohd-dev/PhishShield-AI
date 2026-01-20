import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- PAGE CONFIG ---
st.set_page_config(page_title="PhishShield AI Dashboard", page_icon="🛡️", layout="wide")

# --- CUSTOM CSS FOR STYLING ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    .result-card { padding: 20px; border-radius: 10px; border: 1px solid #ddd; background-color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("phish_model.keras")
    with open('vocab_mapping.pkl', 'rb') as f:
        char2idx = pickle.load(f)
    return model, char2idx

try:
    model, char2idx = load_assets()
except:
    st.error("Model files not found. Please run the training script first.")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2092/2092663.png", width=100)
    st.title("Model Insights")
    st.info("This AI uses a **Long Short-Term Memory (LSTM)** architecture to analyze URL sequences.")
    st.metric(label="Model Accuracy", value="87.12%")
    st.metric(label="Sequence Length", value="40 Chars")
    st.divider()
    st.write("Created by: Mohammed Nabeel")

# --- MAIN UI ---
st.title("🛡️ PhishShield: Deep Learning URL Guard")
st.write("Enter a URL below to analyze its character patterns for phishing signatures.")

col1, col2 = st.columns([2, 1])

with col1:
    url_input = st.text_input("🔗 URL to Scan", placeholder="e.g., bank-secure-login-update.com")
    
    if st.button("Run Security Scan"):
        if url_input:
            # Preprocessing
            clean_url = url_input.replace("https://","").replace("http://","").split("/")[0]
            encoded = [char2idx[c] for c in clean_url if c in char2idx]
            padded = pad_sequences([encoded], maxlen=40)
            
            # Prediction
            with st.spinner('🔍 AI is analyzing character patterns...'):
                prediction = model.predict(padded)[0][0]
            
            # Display Results
            st.subheader("Analysis Results")
            if prediction > 0.5:
                st.error(f"### 🚩 PHISHING DETECTED")
                st.progress(int(prediction * 100))
                st.write(f"The AI is **{prediction*100:.1f}% confident** this URL is malicious.")
            else:
                st.success(f"### ✅ DOMAIN APPEARS SAFE")
                st.progress(int(prediction * 100))
                st.write(f"The AI is **{(1-prediction)*100:.1f}% confident** this URL is legitimate.")
        else:
            st.warning("Please enter a URL to start.")

with col2:
    st.subheader("How it works")
    st.write("""
    1. **Character Tokenization**: The URL is broken into numerical codes.
    2. **Temporal Analysis**: The LSTM scans for suspicious sequences (e.g., 'secure-', '-login', unusual TLDs).
    3. **Probability Scoring**: A sigmoid function outputs a risk factor between 0 and 1.
    """)
    if url_input and 'encoded' in locals():
        st.write("**URL Tokens:**")
        st.caption(f"{encoded}")