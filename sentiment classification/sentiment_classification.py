"""
Sentiment Classification on a Custom Dataset (IMDB Movie Reviews)
-------------------------------------------------------------------
Pipeline:
  1. Load & clean the dataset
  2. Preprocess text (lowercasing, HTML/punctuation removal, tokenization,
     stopword removal, lemmatization) using nltk
  3. Baseline sentiment via TextBlob (no training required)
  4. Vectorize text with TF-IDF (traditional word representation)
  5. Train Naive Bayes, SVM (LinearSVC), and Random Forest classifiers
  6. Evaluate all models: Accuracy, Precision, Recall, F1-score
  7. Visualize model comparison (bar chart) and confusion matrices

Run:
    python sentiment_classification.py

Outputs are written to the `outputs/` folder:
    - model_comparison.csv
    - model_comparison.png
    - confusion_matrices.png
    - classification_reports.txt
"""

import re
import string
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from textblob import TextBlob

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
DATA_PATH = "IMDB_Dataset.csv"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# The full IMDB dataset has 50,000 reviews. To keep training/evaluation fast
# for this exercise (especially SVM / RandomForest), we sample a subset.
# Set SAMPLE_SIZE = None to use the full dataset.
SAMPLE_SIZE = 8000
RANDOM_STATE = 42

# --------------------------------------------------------------------------
# NLTK setup
# --------------------------------------------------------------------------
for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(
            f"tokenizers/{pkg}" if "punkt" in pkg else f"corpora/{pkg}"
        )
    except LookupError:
        nltk.download(pkg, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


# --------------------------------------------------------------------------
# 1 & 2. Load and preprocess
# --------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Lowercase, strip HTML tags/punctuation/numbers."""
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)                 # remove HTML tags (e.g. <br />)
    text = re.sub(r"http\S+|www\S+", " ", text)         # remove URLs
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)                    # remove digits
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess(text: str) -> str:
    """Full pipeline: clean -> tokenize -> remove stopwords -> lemmatize."""
    cleaned = clean_text(text)
    tokens = word_tokenize(cleaned)
    tokens = [
        LEMMATIZER.lemmatize(tok)
        for tok in tokens
        if tok not in STOP_WORDS and len(tok) > 2
    ]
    return " ".join(tokens)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["review", "sentiment"]).drop_duplicates(subset=["review"])
    if SAMPLE_SIZE is not None and len(df) > SAMPLE_SIZE:
        per_class = SAMPLE_SIZE // 2
        parts = [
            group.sample(per_class, random_state=RANDOM_STATE)
            for _, group in df.groupby("sentiment")
        ]
        df = pd.concat(parts).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})
    return df


# --------------------------------------------------------------------------
# 3. Baseline: TextBlob polarity-based sentiment
# --------------------------------------------------------------------------
def textblob_predict(text: str) -> int:
    polarity = TextBlob(text).sentiment.polarity
    return 1 if polarity >= 0 else 0


# --------------------------------------------------------------------------
# 5 & 6. Train models and evaluate
# --------------------------------------------------------------------------
def evaluate(name, y_true, y_pred, reports: dict) -> dict:
    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
    }
    reports[name] = classification_report(y_true, y_pred, target_names=["negative", "positive"])
    return metrics


def main():
    print("Loading and preprocessing data...")
    df = load_data()
    print(f"Dataset size after sampling: {len(df)} reviews "
          f"({df['label'].value_counts().to_dict()})")

    df["clean_review"] = df["review"].apply(preprocess)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_review"], df["label"],
        test_size=0.2, random_state=RANDOM_STATE, stratify=df["label"]
    )

    results = []
    reports = {}

    # ---- Baseline: TextBlob (no training needed) ----
    print("Running TextBlob baseline...")
    tb_preds = X_test.apply(textblob_predict)
    results.append(evaluate("TextBlob (baseline)", y_test, tb_preds, reports))

    # ---- Traditional word representation: TF-IDF ----
    print("Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    models = {
        "Naive Bayes": MultinomialNB(),
        "SVM (LinearSVC)": LinearSVC(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    predictions = {"TextBlob (baseline)": tb_preds}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_tfidf, y_train)
        preds = model.predict(X_test_tfidf)
        predictions[name] = preds
        results.append(evaluate(name, y_test, preds, reports))

    # ---- Save metrics table ----
    results_df = pd.DataFrame(results).set_index("Model").round(4)
    print("\nModel comparison:\n", results_df)
    results_df.to_csv(OUTPUT_DIR / "model_comparison.csv")

    with open(OUTPUT_DIR / "classification_reports.txt", "w") as f:
        for name, rep in reports.items():
            f.write(f"=== {name} ===\n{rep}\n\n")

    # ---- Visualization 1: bar chart comparing metrics ----
    ax = results_df.plot(kind="bar", figsize=(10, 6), rot=20)
    ax.set_title("Model Comparison: Sentiment Classification")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=150)
    plt.close()

    # ---- Visualization 2: confusion matrices ----
    fig, axes = plt.subplots(1, len(predictions), figsize=(5 * len(predictions), 4))
    for ax, (name, preds) in zip(axes, predictions.items()):
        cm = confusion_matrix(y_test, preds)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["neg", "pos"])
        ax.set_yticklabels(["neg", "pos"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=150)
    plt.close()

    print(f"\nAll outputs saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
