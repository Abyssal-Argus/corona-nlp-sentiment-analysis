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

nltk.download("stopwords")
stop_words = set(stopwords.words("english") + list(string.punctuation))

X = []
y = []

with open("sample_data/Corona_NLP_Test.csv", "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    print("CSV columns:", reader.fieldnames)

    for row in reader:
        tweet = row["OriginalTweet"]
        sentiment = row["Sentiment"]

        if tweet and sentiment:
            X.append(tweet)
            y.append(sentiment)

print("Total samples:", len(X))

unique_labels = sorted(set(y))
label_to_num = {label: idx for idx, label in enumerate(unique_labels)}
print("Label mapping:", label_to_num)

y_num = np.array([label_to_num[label] for label in y], dtype=int)
X = np.array(X, dtype=object)

np.random.seed(42)
indices = np.arange(len(X))
np.random.shuffle(indices)

X = X[indices]
y_num = y_num[indices]

split_index = int(0.8 * len(X))
X_train = X[:split_index]
X_test = X[split_index:]
y_train = y_num[:split_index]
y_test = y_num[split_index:]

X_train_raw = X_train.copy()
X_test_raw = X_test.copy()

print("Train size:", len(X_train))
print("Test size:", len(X_test))
print("Example X_train, y_train:", X_train[:3], y_train[:3])
print("Labels:", unique_labels)

def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

X_train_clean = np.array([clean_text(text) for text in X_train_raw], dtype=object)
X_test_clean = np.array([clean_text(text) for text in X_test_raw], dtype=object)

print("\nExample cleaned training samples:")
for i in range(3):
    print(f"Original: {X_train_raw[i]}")
    print(f"Cleaned : {X_train_clean[i]}")
    print()

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train_clean)
X_test_vec = vectorizer.transform(X_test_clean)

print("X_train_vec shape:", X_train_vec.shape)
print("X_test_vec shape:", X_test_vec.shape)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=unique_labels))