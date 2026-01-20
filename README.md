# 🛡️ PhishShield: Deep Learning URL Guard

An AI-powered phishing detection system built with **TensorFlow** and **Streamlit**. This project uses a **Long Short-Term Memory (LSTM)** Recurrent Neural Network to analyze the character-level patterns of URLs and identify malicious intent.

## 📊 Performance
- **Accuracy:** 87.12%
- **Model Type:** RNN (LSTM)
- **Input:** Raw URL string (Top-level domain)

## 🛠️ How It Works
The model processes URLs as sequences of characters. 
1. **Tokenization**: Converts characters into numerical values based on a learned vocabulary.
2. **LSTM Layers**: Captures temporal dependencies (e.g., suspicious keyword sequences like `-update-login`).
3. **Sigmoid Output**: Predicts a probability score between 0 (Safe) and 1 (Phishing).

## 📥 Installation & Usage
1. Clone the repo: `git clone https://github.com/yourusername/PhishShield-AI.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the App: `streamlit run app.py`

## 👤 Author
**Mohammed Nabeel**
Master's Student | University of Limerick
