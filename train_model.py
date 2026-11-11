import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from xgboost import XGBClassifier

import joblib


fake_df = pd.read_csv("Fake.csv.xls")
fake_df["label"] = "FAKE"

true_df = pd.read_csv("True.csv.xls")
true_df["label"] = "REAL"

df = pd.concat([fake_df, true_df], ignore_index=True)
print("Combined dataset shape:", df.shape)
print("\nClass distribution:")
print(df["label"].value_counts())

df= df.drop_duplicates()
df=df.reset_index(drop=True)

df["content"]= df["title"]+ " "+ df["text"]

df=df.drop_duplicates(subset="content", keep="first")

df=df.reset_index(drop=True)

print("\nCleaned dataset shape:", df.shape)
print("\nFinal class distribution:")
print(df["label"].value_counts())

X = df["content"]
Y = df["label"]

X_train, X_test, Y_train, Y_test = train_test_split(
  X,
  Y,
  test_size=0.2,
  random_state=42,
  stratify=Y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

tfidf = TfidfVectorizer(max_features=20000)

X_train_tfidf = tfidf.fit_transform(X_train)

X_test_tfidf = tfidf.transform(X_test)

print("\nTraining TF-IDF shape:", X_train_tfidf.shape)
print("Testing TF-IDF shape:", X_test_tfidf.shape)

Y_train_xgb = Y_train.map({
    "FAKE": 0,
    "REAL": 1
})

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss"
)

xgb_model.fit(
    X_train_tfidf,
    Y_train_xgb
)

print("\nXGBoost evaluation model trained.")

xgb_pred_numeric = xgb_model.predict(X_test_tfidf)

xgb_pred = pd.Series(xgb_pred_numeric).map({
    0: "FAKE",
    1: "REAL"
})

xgb_accuracy = accuracy_score(Y_test, xgb_pred)

print("\nXGBoost Accuracy:", xgb_accuracy)
print("XGBoost Accuracy (%):", xgb_accuracy * 100)

print("\nXGBoost Classification Report:")
print(classification_report(Y_test, xgb_pred))

print("\nXGBoost Confusion Matrix:")
print(confusion_matrix(Y_test, xgb_pred))


# ==========================================
# FINAL MODEL TRAINING
# ==========================================

final_tfidf = TfidfVectorizer(max_features=20000)

X_final_tfidf = final_tfidf.fit_transform(X)

print("\nFinal TF-IDF shape:", X_final_tfidf.shape)


Y_final_xgb = Y.map({
    "FAKE": 0,
    "REAL": 1
})

print("\nFinal training labels:")
print(Y_final_xgb.value_counts())


final_xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss"
)


final_xgb_model.fit(
    X_final_tfidf,
    Y_final_xgb
)

print("\nFinal XGBoost model trained successfully.")

import os
os.makedirs("models", exist_ok=True)

# ==========================================
# SAVE FINAL MODEL
# ==========================================

joblib.dump(
    final_xgb_model,
    "models/final_xgb_model.pkl"
)

joblib.dump(
    final_tfidf,
    "models/final_tfidf.pkl"
)

print("\nFinal model and TF-IDF vectorizer saved successfully.")