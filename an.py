import re
import string
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


def clean_text(text: str):
    text = str(text).lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text
df = pd.read_csv(r"D:\Downloads\tech_minds\raw_data\sentiment_dataset.csv")

df = df[["text", "label"]].dropna()
df["text"] = df["text"].apply(clean_text)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["label"])
X = df["text"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Naive Bayes": MultinomialNB(),
    "SVM": LinearSVC()
}

best_model = None
best_score = 0

for name, model in models.items():
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(f"\n{name}")
    print("Accuracy:", round(acc, 4))
    print(classification_report(y_test, y_pred))

    if acc > best_score:
        best_score = acc
        best_model = pipeline

joblib.dump(best_model, "best_sentiment_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("\nBest model saved successfully!")