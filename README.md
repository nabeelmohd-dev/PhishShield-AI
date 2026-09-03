# 🛡️ PhishShield: Deep Learning URL Guard

An AI-powered phishing detection system built with **TensorFlow** and **Streamlit**. This project uses a **Long Short-Term Memory (LSTM)** recurrent neural network to analyze the character-level patterns of URLs and identify malicious intent, served through an interactive web dashboard.

## 📊 Performance

- **Accuracy:** 87.12% *(reported from prior training run, re-verify if the training data or preprocessing has changed since)*
- **Model Type:** RNN (LSTM)
- **Input:** Raw URL string, reduced to its domain name (e.g. `login-verify-secure.xyz`)
- **Sequence Length:** 40 characters

## 🛠️ How It Works

The model processes URLs as sequences of characters.

1. **Tokenization:** Converts characters into numerical values based on a learned vocabulary built from the training data.
2. **LSTM Layer:** Captures sequential dependencies in the domain string (e.g. suspicious keyword patterns like `-update-login`).
3. **Sigmoid Output:** Predicts a probability score between 0 (safe) and 1 (phishing), shown in the dashboard with a confidence percentage.

Training data is pulled automatically on first run from two public sources: a labelled phishing URL feed and the Cisco Umbrella top-1-million domains list, used as the pool of benign examples. Benign domains are oversampled relative to the phishing set (configurable via `OVERSAMPLING_RATE`), with class weights applied during training to correct for the imbalance.

## 📁 Project Structure

* `train.py`  downloads training data, builds the vocabulary, trains the LSTM, and (see note below) must save the trained model and vocabulary for the dashboard to use.
* `app.py`  the Streamlit dashboard. Loads the saved model and vocabulary, and lets you paste in a URL for a live prediction.

## ⚠️ Setup Note: Training Script Must Save Its Outputs

The dashboard (`app.py`) expects two files in the project root:

* `phish_model.keras`  the trained Keras model
* `vocab_mapping.pkl`  the pickled `char2idx` vocabulary dictionary used during training

The training script does not currently write either of these to disk on its own, it trains the model and prints a couple of example predictions to the console, but nothing is saved. Before the dashboard will run, add something like this to the end of the training script:

```python
model.save("phish_model.keras")
with open("vocab_mapping.pkl", "wb") as f:
    pickle.dump(char2idx, f)
```

(`pickle` needs to be imported in the training script too.)

## 📥 Installation & Usage

### Prerequisites

* Python 3.9 or higher
* Internet access on first run (the training script downloads its own data via `curl`)

### Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/PhishShield-AI.git
   cd PhishShield-AI
   ```
2. Install dependencies:
   ```bash
   pip install tensorflow streamlit pandas numpy scikit-learn prettytable matplotlib
   ```
3. Train the model (see the setup note above, this must save `phish_model.keras` and `vocab_mapping.pkl`):
   ```bash
   python train.py
   ```
4. Launch the dashboard:
   ```bash
   streamlit run app.py
   ```

Paste a URL into the input box and click **Run Security Scan** to get a live phishing prediction with a confidence score.

## 👤 Author

**Mohammed Nabeel**
Master's Student | University of Limerick
