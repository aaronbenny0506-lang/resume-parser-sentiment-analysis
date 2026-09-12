# NLP Mini-Projects: Resume Extractor & Sentiment Classification

This repository contains two independent NLP tasks:

1. **Resume Extractor** : a rule-based script using regular expressions to
   extract phone numbers and email addresses from resume text.
2. **Sentiment Classification** : a text classification pipeline that
   trains and compares multiple ML models (Naive Bayes, SVM, Random
   Forest) against a TextBlob baseline on the IMDB movie review dataset.

---

## 📁 Repository Structure

```
.
├── resume_extractor/
│   ├── resume_extractor.py     # Main script (regex extraction + sample tests)
│   └── output.json             # Generated structured output
│
├── sentiment_classification/
│   ├── sentiment_classification.py    # End-to-end script version
│   ├── sentiment_classification.ipynb # Jupyter notebook version (with outputs)
│   ├── IMDB_Dataset.csv               # Dataset (50,000 labeled movie reviews)
│   └── outputs/
│       ├── model_comparison.csv
│       ├── model_comparison.png
│       ├── confusion_matrices.png
│       └── classification_reports.txt
│
├── requirements.txt
└── README.md
```

---

## 🎯 Objective 1: Resume Contact Info Extractor

`resume_extractor/resume_extractor.py` uses two regular expressions:

- **Email regex** — matches standard `local-part@domain.tld` addresses.
- **Phone regex** — handles common formats including country codes,
  parenthesized area codes, and separators (`-`, `.`, space), eg:
  - `+1 (555) 123-4567`
  - `555-987-6543`
  - `+91 98765 43210`
  - `(022) 2345 6789`
  - `07911.123456`

A lightweight validity check (7–15 digit count) filters out obvious
non-phone-number matches (dates, zip codes, etc.).

### Run it

```bash
cd resume_extractor
python resume_extractor.py
```

This runs the extractor against **3 built-in sample resumes** and writes
the results to `output.json`, eg:

```json
{
  "resume_1.txt": {
    "emails": ["john.doe1990@gmail.com", "john.doe.work@acme.com"],
    "phone_numbers": ["+1 (555) 123-4567", "555-987-6543"]
  },
  "resume_2.txt": {
    "emails": ["priya.sharma@outlook.com"],
    "phone_numbers": ["+91 98765 43210", "(022) 2345 6789"]
  },
  "resume_3.txt": {
    "emails": ["michael.obrien+jobs@company.co.uk", "m.obrien@yahoo.com"],
    "phone_numbers": ["20 7946 0958", "07911.123456", "555.222.3333"]
  }
}
```

You can also import the functions directly for your own text:

```python
from resume_extractor import extract_contact_info

info = extract_contact_info(open("my_resume.txt").read())
print(info)
```

---

## 🎯 Objective 2: Sentiment Classification

`sentiment_classification/` contains both a `.py` script and an executed
`.ipynb` notebook implementing the full pipeline on the **IMDB Movie
Reviews dataset** (50,000 labeled reviews, balanced positive/negative).

> Note: the task referenced a Google Drive-hosted sentiment dataset. The
> IMDB dataset included here (`IMDB_Dataset.csv`) is used as the custom
> sentiment dataset, swap in any other CSV with `review`/`sentiment`
> columns and the pipeline works unchanged.

### Pipeline

1. **Preprocessing** (`nltk`): lowercasing, HTML/URL/punctuation/digit
   removal, tokenization, stopword removal, lemmatization.
2. **Baseline**: `TextBlob` polarity-based sentiment (no training).
3. **Text representation**: `TfidfVectorizer` (unigrams + bigrams, top
   5,000 features).
4. **Models trained**:
   - Multinomial Naive Bayes (`sklearn`)
   - Linear SVM (`LinearSVC`)
   - Random Forest
5. **Evaluation**: Accuracy, Precision, Recall, F1-score for every model.
6. **Visualizations**: grouped bar chart comparing all metrics across
   models, plus side-by-side confusion matrices.

### Run it

```bash
cd sentiment_classification
python sentiment_classification.py
```

Or open `sentiment_classification.ipynb` in Jupyter to see the full
walkthrough with inline outputs.

Outputs are written to `sentiment_classification/outputs/`:
- `model_comparison.csv` : metrics table
- `model_comparison.png` : bar chart of all models
- `confusion_matrices.png` : confusion matrix per model
- `classification_reports.txt` : full sklearn classification reports

### Sample Results (8,000-review balanced sample)

| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|---------:|----------:|-------:|---------:|
| TextBlob (baseline) |   0.70   |   0.64    |  0.93  |   0.76   |
| Naive Bayes         |   0.87   |   0.86    |  0.88  |   0.87   |
| SVM (LinearSVC)     |   0.85   |   0.85    |  0.84  |   0.85   |
| Random Forest       |   0.84   |   0.84    |  0.83  |   0.84   |

*(Naive Bayes and SVM both comfortably outperform the untrained TextBlob
baseline; results may vary slightly by random seed / sample.)*

> By default the script samples 8,000 reviews (4,000 per class) to keep
> training fast. Set `SAMPLE_SIZE = None` in the script/notebook to train
> on the full 50,000-review dataset (slower, especially for Random Forest).

---

## ⚙️ Setup

```bash
pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab stopwords wordnet omw-1.4
```

## 📦 Requirements

See [`requirements.txt`](./requirements.txt).
