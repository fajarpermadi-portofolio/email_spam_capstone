#!/usr/bin/env python
# coding: utf-8

# ## IMPORT LIBRARY

# In[1]:


import re
import string
import joblib
import numpy as np
import pandas as pd
import hashlib
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import skfuzzy as fuzz
from skfuzzy import control as ctrl


# ## NLP Tools

# In[2]:


nltk.download('stopwords')
stop_words = set(stopwords.words('indonesian'))
factory = StemmerFactory()
stemmer = factory.create_stemmer()


# ## DAFTAR KATA PENTING

# In[3]:


BLOOM_VERBS = [
    "analyze", "evaluate", "create", "design", "develop", "assess",
    "compare", "apply", "interpret", "explain", "demonstrate",
    "construct", "formulate", "summarize", "predict"
]

THEMATIC_TERMS = [
    "account", "verify", "password", "login", "transaction", "bank",
    "security", "urgent", "click", "payment", "offer", "bonus",
    "transfer", "discount", "alert", "update", "confirm"
]

SUPPORTING_VERBS = [
    "check", "verify", "click", "read", "reply", "share",
    "follow", "install", "open", "download"
]


# ## Preprocessing & Pembobotan Bloom

# In[4]:


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

def bloom_weight(text):
    words = text.split()
    score = 0
    for w in words:
        if w in BLOOM_VERBS:
            score += 5
        elif w in THEMATIC_TERMS:
            score += 4
        elif w in SUPPORTING_VERBS:
            score += 3
        else:
            score += 1
    return score / max(1, len(words))


# ## Fuzzy Logic untuk Penilaian Risiko

# In[5]:


def fuzzy_weight(bloom_score):
    risk = ctrl.Antecedent(np.arange(0, 6, 0.1), 'risk')
    weight = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'weight')

    risk['low'] = fuzz.trimf(risk.universe, [0, 0, 2])
    risk['medium'] = fuzz.trimf(risk.universe, [1, 3, 5])
    risk['high'] = fuzz.trimf(risk.universe, [3, 5, 5])

    weight['low'] = fuzz.trimf(weight.universe, [0, 0, 0.5])
    weight['medium'] = fuzz.trimf(weight.universe, [0.3, 0.6, 0.9])
    weight['high'] = fuzz.trimf(weight.universe, [0.7, 1, 1])

    rule1 = ctrl.Rule(risk['low'], weight['low'])
    rule2 = ctrl.Rule(risk['medium'], weight['medium'])
    rule3 = ctrl.Rule(risk['high'], weight['high'])

    control = ctrl.ControlSystem([rule1, rule2, rule3])
    sim = ctrl.ControlSystemSimulation(control)
    sim.input['risk'] = bloom_score
    sim.compute()
    return sim.output['weight']


# ## Load Dataset

# In[6]:


def load_dataset():
    df1 = pd.read_csv("email_spam_indo.csv")
    df2 = pd.read_csv("spam.csv", encoding_errors='ignore')

    df1.columns = [c.lower() for c in df1.columns]
    df2.columns = [c.lower() for c in df2.columns]

    text_col = [c for c in df1.columns if 'pesan' in c or 'message' in c][0]
    label_col = [c for c in df1.columns if 'kategori' in c or 'label' in c][0]

    df = pd.concat([
        df1[[text_col, label_col]],
        df2[[df2.columns[1], df2.columns[0]]].rename(columns={df2.columns[1]: text_col, df2.columns[0]: label_col})
    ], ignore_index=True)

    df["clean"] = df[text_col].apply(clean_text)
    df["bloom"] = df["clean"].apply(bloom_weight)
    df["fuzzy"] = df["bloom"].apply(fuzzy_weight)

    print("✅ Dataset berhasil dimuat:", df.shape)
    return df, text_col, label_col


# ## Pipeline Training Multi-Model

# In[7]:


def train_models(df, text_col, label_col):
    vectorizer = TfidfVectorizer(max_features=3000)
    X = vectorizer.fit_transform(df["clean"])
    y = df[label_col].astype(str)

    models = {
        "Naive Bayes": MultinomialNB(),
        "SVM": SVC(kernel="linear", probability=True),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "DNN": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300)
    }

    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5)
        results[name] = np.mean(scores)
        print(f"🔹 {name} | K-Fold Acc: {scores.mean():.4f}")

    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name].fit(X, y)

    y_pred = best_model.predict(X)
    print("\n📊 Confusion Matrix:")
    print(confusion_matrix(y, y_pred))
    print("\n📈 Classification Report:")
    print(classification_report(y, y_pred))

    joblib.dump(best_model, "best_model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
    print(f"🏆 Model terbaik: {best_model_name} ({results[best_model_name]:.4f})")
    return best_model, vectorizer, results


# ## Kriptografi Ringan (Hash Validasi)

# In[8]:


def generate_hash(prediction):
    return hashlib.sha256(prediction.encode()).hexdigest()


# ## Prediksi Email Baru

# In[9]:


def predict_email(text, model, vectorizer):
    cleaned = clean_text(text)
    X_input = vectorizer.transform([cleaned])
    pred = model.predict(X_input)[0]
    bloom = bloom_weight(cleaned)
    fuzzy_score = fuzzy_weight(bloom)
    hash_value = generate_hash(pred)
    return pred, bloom, fuzzy_score, hash_value


# ## Main Flow (Testing Pipeline)

# In[10]:


if __name__ == "__main__":
    df, text_col, label_col = load_dataset()
    model, vectorizer, results = train_models(df, text_col, label_col)

    test_text = "Selamat! Anda memenangkan hadiah Rp50 juta! Klik tautan berikut."
    pred, bloom, fuzzy, hashv = predict_email(test_text, model, vectorizer)

    print("\n=== Hasil Uji ===")
    print("Prediksi:", pred)
    print("Bloom Score:", round(bloom, 2))
    print("Fuzzy Risk:", round(fuzzy, 2))
    print("SHA256 Hash:", hashv[:20], "...")

