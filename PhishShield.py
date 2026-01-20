import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from prettytable import PrettyTable
from collections import defaultdict

# --- 1. CONFIGURATION & DATA ---
RANDOM_SEED = 16
np.random.seed(RANDOM_SEED)
MAX_SEQ_LEN = 40
OVERSAMPLING_RATE = 1.5

def download_data():
    files = {
        "combined_online_valid.csv": "https://raw.githubusercontent.com/maximsachs/phishing_classification_recurrent_nn/master/combined_online_valid.csv",
        "top-1m_umbrella.csv": "https://raw.githubusercontent.com/maximsachs/phishing_classification_recurrent_nn/master/top-1m_umbrella.csv"
    }
    for name, url in files.items():
        if not os.path.exists(name):
            print(f"Downloading {name}...")
            os.system(f"curl -O {url}")

# --- 2. PREPROCESSING FUNCTIONS ---
def get_char_mapping(texts):
    vocab = sorted(set("".join(texts)), reverse=True)
    vocab.insert(0, " ")  # Padding character
    char2idx = {u: i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)
    return char2idx, idx2char, len(vocab)

def text_to_int(text, char2idx):
    return np.array([char2idx[c] for c in text if c in char2idx])

# --- 3. EVALUATION TOOLS ---
def statistics_evaluator(predictions_binary, y_true):
    # 00:TN, 01:FP, 10:FN, 11:TP
    hypothesis_tests = [int(str(label)+str(pred), 2) for pred, label in zip(predictions_binary, y_true)]
    unique, counts = np.unique(hypothesis_tests, return_counts=True)
    counts_dict = dict(zip(unique, counts))
    
    res = {
        "TN": counts_dict.get(0, 0),
        "FP": counts_dict.get(1, 0),
        "FN": counts_dict.get(2, 0),
        "TP": counts_dict.get(3, 0)
    }
    return res

# --- 4. MAIN EXECUTION FLOW ---
def main():
    download_data()

    # Load and clean data
    online_valid_df = pd.read_csv("combined_online_valid.csv")
    online_valid_df["domain_names"] = online_valid_df["url"].str.replace("https://|http://", "", regex=True).str.split("/").str[0]
    
    whitelist_df = pd.read_csv("top-1m_umbrella.csv", header=None, names=["rank", "domain_names"])
    
    # Filter intersections
    common = np.intersect1d(online_valid_df["domain_names"], whitelist_df["domain_names"])
    phishing = online_valid_df[~online_valid_df["domain_names"].isin(common)]["domain_names"].values
    safe_pool = whitelist_df[~whitelist_df["domain_names"].isin(common)]["domain_names"].values
    benign = np.random.choice(safe_pool, size=int(OVERSAMPLING_RATE * len(phishing)), replace=False)

    # Combine datasets
    X = list(phishing) + list(benign)
    y = [1]*len(phishing) + [0]*len(benign)
    weights = [1]*len(phishing) + [1/OVERSAMPLING_RATE]*len(benign)

    # Encoding
    char2idx, idx2char, vocab_size = get_char_mapping(X)
    X_encoded = [text_to_int(d, char2idx) for d in X]
    X_padded = sequence.pad_sequences(X_encoded, maxlen=MAX_SEQ_LEN)

    # Split
    X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
        X_padded, np.array(y), np.array(weights), test_size=0.15, random_state=RANDOM_SEED
    )

    # Model definition
    model = Sequential([
        Embedding(vocab_size, 64, input_length=MAX_SEQ_LEN),
        LSTM(128),
        Dense(128, activation="tanh"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
    
    # Train
    class_weight = {0: (1/(OVERSAMPLING_RATE+1)), 1: (OVERSAMPLING_RATE/(OVERSAMPLING_RATE+1))}
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    
    print("\nStarting Training...")
    model.fit(X_train, 
              y_train, 
              epochs=10, 
              validation_data=(X_test, y_test), 
              class_weight=class_weight, 
              callbacks=[early_stop])

    # Final Prediction Example
    def quick_predict(url):
        enc = sequence.pad_sequences([text_to_int(url, char2idx)], maxlen=MAX_SEQ_LEN)
        prob = model.predict(enc)[0][0]
        print(f"URL: {url} | Result: {'PHISHING' if prob > 0.5 else 'SAFE'} ({prob:.4f})")

    print("\nTesting Model:")
    quick_predict("google.com")
    quick_predict("login-verify-secure.xyz")

if __name__ == "__main__":
    main()