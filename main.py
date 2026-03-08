import csv
import numpy as np
from bs4 import BeautifulSoup
import re
import nltk
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download("stopwords", quiet=True)

stop_words = set(stopwords.words("english") + list(string.punctuation))

X, y = [], []

try:
    with open("sample_data/Corona_NLP_Test.csv", "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get("OriginalTweet") and row.get("Sentiment"):
                X.append(row["OriginalTweet"])
                y.append(row["Sentiment"])
    
    if not X:
        raise ValueError("No data loaded")
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

print(f"Loaded {len(X)} samples")

unique_labels = sorted(set(y))
label_to_num = {label: idx for idx, label in enumerate(unique_labels)}
print("Labels:", label_to_num)

y_num = np.array([label_to_num[label] for label in y])
np.random.seed(42)
indices = np.arange(len(X))
np.random.shuffle(indices)
X = np.array(X)[indices]
y_num = y_num[indices]

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y_num[:split], y_num[split:]
X_train_raw, X_test_raw = X_train.copy(), X_test.copy()

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

def clean_text(text):
    try:
        text = BeautifulSoup(str(text), "html.parser").get_text().lower()
        text = re.sub(r"http\S+|www\S+|\d+", "", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        return " ".join([w for w in text.split() if w not in stop_words])
    except:
        return ""

try:
    X_train_clean = [clean_text(t) for t in X_train_raw]
    X_test_clean = [clean_text(t) for t in X_test_raw]
    
    print("\nCleaning examples:")
    for i in range(min(2, len(X_train))):
        print(f"Original: {X_train_raw[i][:80]}...")
        print(f"Cleaned: {X_train_clean[i][:80]}...\n")
    
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train_clean)
    X_test_vec = vectorizer.transform(X_test_clean)
    
    print(f"Vectors: {X_train_vec.shape}, {X_test_vec.shape}")
    
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=unique_labels))
    
except Exception as e:
    print(f"Error: {e}")
    exit(1)

print("\n" + "="*50)
print("Process completed successfully!")
print("="*50)