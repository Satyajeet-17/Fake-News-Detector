import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
from xgboost import XGBClassifier



fake_df=pd.read_csv("fake.csv")
fake_df["label"]="FAKE"
#print(fake_df.head())
#print(fake_df.shape)
#print(fake_df.columns)
#print(fake_df.info())
#print(fake_df["title"].head())
#print(fake_df["subject"].head())
#print(fake_df[["title","subject"]].head())
#print(fake_df[["title","subject"]].shape)
#print(fake_df["subject"].unique())
#print(fake_df["subject"].value_counts())



true_df = pd.read_csv("True.csv")
true_df["label"]="REAL"
#print(true_df.head())
#print(true_df.shape)
#print(true_df.columns)
#true_df.info()

df= pd.concat([fake_df, true_df], ignore_index=True)
print(df.shape)
#print(df.head())
print(df.index)
print(df["label"].value_counts())
print("Duplicate rows: ", df.duplicated().sum())
print("Duplicate texts",df["text"].duplicated().sum())
print("\n Missing values:")
print(df.isnull().sum())
text_label_counts=df.groupby("text")["label"].nunique()
print("Text appearing with different labels: ", (text_label_counts>1).sum())

#conflicting_texts= text_label_counts[text_label_counts>1].index
#print(df[df["text"].isin(conflicting_texts)][
#  ["title","text","subject","date","label"]
#  ].to_string(index=False))

empty_text= df["text"].str.strip().eq("")
#print("Empty text rows:", empty_text.sum())
#
empty_title= df["title"].str.strip().eq("")
#print("Empty title rows:", empty_title.sum())
#
duplicated_texts= df["text"].duplicated(keep=False)
#print("Rows belonging to duplicated texts:", duplicated_texts.sum())

non_empty_text= df["text"].str.strip().ne("")
duplicate_non_empty_text = df["text"].duplicated(keep=False) & non_empty_text
#print("Non-empty rows belonging to duplicated texts:", duplicate_non_empty_text.sum())
#
non_empty_df = df[non_empty_text]
#text_label_counts_non_empty = non_empty_df.groupby("text")["label"].nunique()
#print(
#  "Non-empty texts appearing with different labels:", (text_label_counts_non_empty>1).sum()
#)
#
#print("Before removing exact duplicates:", df.shape)
df= df.drop_duplicates()
#print("After removing exact duplicates:", df.shape)
#
df=df.reset_index(drop=True)
#print(df.index)
#print(df.shape)
#
#empty_text= df["text"].str.strip().eq("")
#print("Empty text rows:", empty_text.sum())
#
df['content']= df['title']+" "+ df["text"]
##print(df[["title","text","content"]].head())
#print("Empty content rows:", df["content"].str.strip().eq("").sum())
#
duplicate_content = df["content"].duplicated(keep=False)
#print("Rows belonging to duplicated content: ", duplicate_content.sum())
#
#print("Total rows:", len(df))
#print("Unique contents:", df["content"].nunique())
#
df= df.drop_duplicates(subset='content', keep='first')
df=df.reset_index(drop=True)


#print("Shape after content deduplication:",df.shape)
#print("Unique contents:", df["content"].nunique())
#print(df["label"].value_counts())

#numbers= np.array([10,20,30,40])
#
#print(numbers)
#print(numbers.shape)
#
#numbers_2d= np.array([
#  [10,20,30],
#  [40,50,60]
#])
#
#print(numbers_2d)
#print(numbers_2d.shape)
#
#print(numbers_2d[0])
#print(numbers_2d[0,2])
#print(numbers_2d[:,1])
#
#print(numbers*2)
#print(numbers+5)

X = df["content"]
Y =df["label"]

#print("X shape:", X.shape)
#print("Y shape:", Y.shape)

#print("\n First X value: ")
#print(X.iloc[0])

#print("\n First Y value: ")
#print(Y.iloc[0])

X_train, X_test, Y_train, Y_test= train_test_split(
  X,
  Y,
  test_size=0.2,
  random_state=42,
  stratify=Y
)

#print("X_train:", X_train.shape)
#print("X_test:",X_test.shape)
#
#print("Y_train:", Y_train.shape)
#print("Y_test:",Y_test.shape)
#
#print("Training labels:")
#print(Y_train.value_counts())
#
#print("\nTesting labels:")
#print(Y_test.value_counts())


tfidf= TfidfVectorizer()
X_train_tfidf= tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("X_train TF-IDF shape:", X_train_tfidf.shape)
print("X_test TF-IDF shape:", X_test_tfidf.shape)


#####      LOGISTIC REGRESSION
#
#model= LogisticRegression(max_iter=1000)
#model.fit(X_train_tfidf, Y_train)
#print("Logistic regression model training completed!")
#
#y_pred= model.predict(X_test_tfidf)
#
#print("Number of predictions:", len(y_pred))
#print("First 10 predictions of Logistic regression:", y_pred[:10])
#
#print("Actual labels:", Y_test.iloc[:10].values)
#
#lr_accuracy =accuracy_score(Y_test, y_pred)
#
#print("Logistic regression Accuracy:", lr_accuracy)
#print("Logistic regression Accuracy percentage:", lr_accuracy*100)
#
#cm= confusion_matrix(Y_test, y_pred)
#print(cm)
#
#print(classification_report(Y_test,y_pred))
#
#lr_report= classification_report(
#  Y_test,y_pred, output_dict=True
#)
#print(lr_report["FAKE"]["precision"])
#print(lr_report["FAKE"]["recall"])
#print(lr_report["FAKE"]["f1-score"])



#####        NAIVE BAYES

#nb_model = MultinomialNB()
#nb_model.fit(X_train_tfidf, Y_train)
#print("\nNaive bayes training completed!!")
#
#nb_y_pred=nb_model.predict(X_test_tfidf)
#print("Number of predictions:", len(nb_y_pred))
#print("First 10 predictions of Naive bayes:", nb_y_pred[:10])
#print("Actual nb labels:", Y_test.iloc[:10].values)
#nb_accuracy =accuracy_score(Y_test, nb_y_pred)
#
#print("Naive bayes Accuracy:", nb_accuracy)
#print("Naive bayes Accuracy percentage:", nb_accuracy*100)
#print("Naive Bayes classification report:")
#print(classification_report(Y_test,nb_y_pred))
#
#print("Naive Bayes confusion matrix:")
#print(confusion_matrix(Y_test, nb_y_pred))
#
#nb_report= classification_report(
#  Y_test,nb_y_pred, output_dict=True
#)
#print(nb_report["FAKE"]["precision"])
#print(nb_report["FAKE"]["recall"])
#print(nb_report["FAKE"]["f1-score"])






#####        SVM MODEL

svm_model=LinearSVC()
svm_model.fit(X_train_tfidf, Y_train)
print("\nSVM training completed!!")

svm_y_pred=svm_model.predict(X_test_tfidf)
print("Number of SVM predictions:", len(svm_y_pred))
print("First 10 predictions of SVM:", svm_y_pred[:10])
print("Actual svm labels:", Y_test.iloc[:10].values)
svm_accuracy =accuracy_score(Y_test, svm_y_pred)

print("SVM Accuracy:", svm_accuracy)
print("SVM Accuracy percentage:", svm_accuracy*100)
print("\nSVM classification report:")
print(classification_report(Y_test,svm_y_pred))

print("SVM confusion matrix:")
print(confusion_matrix(Y_test, svm_y_pred))

svm_report= classification_report(
  Y_test,svm_y_pred, output_dict=True
)
print(svm_report["FAKE"]["precision"])
print(svm_report["FAKE"]["recall"])
print(svm_report["FAKE"]["f1-score"])




#####     RANDOM FOREST
#
#rf_model= RandomForestClassifier(
#  n_estimators=100,
#  random_state=42,
#  n_jobs=-1
#)
#rf_model.fit(X_train_tfidf, Y_train)
#
#print("\nRandom forest training completed!!")
#
#rf_y_pred=rf_model.predict(X_test_tfidf)
#print("Number of Random forest predictions:", len(rf_y_pred))
#print("First 10 predictions of Random forest:", rf_y_pred[:10])
#print("Actual random forest labels:", Y_test.iloc[:10].values)
#rf_accuracy =accuracy_score(Y_test, rf_y_pred)
#
#print("Random forest Accuracy:", rf_accuracy)
#print("Random forest Accuracy percentage:", rf_accuracy*100)
#print("\nRandom forest classification report:")
#print(classification_report(Y_test,rf_y_pred))
#
#print("Random forest confusion matrix:")
#print(confusion_matrix(Y_test, rf_y_pred))
#
#rf_report= classification_report(
#  Y_test,rf_y_pred, output_dict=True
#)
#print(rf_report["FAKE"]["precision"])
#print(rf_report["FAKE"]["recall"])
#print(rf_report["FAKE"]["f1-score"])
#
#print("LR FAKE F1:", lr_report["FAKE"]["f1-score"])
#print("NB FAKE F1:", nb_report["FAKE"]["f1-score"])
#print("SVM FAKE F1:", svm_report["FAKE"]["f1-score"])
#print("RF FAKE F1:", rf_report["FAKE"]["f1-score"])
#
#
#results=[
#  {
#    "Model": "Logistic Regression",
#    "Accuracy": lr_accuracy,
#    "FAKE Precision": lr_report["FAKE"]["precision"],
#    "FAKE Recall": lr_report["FAKE"]["recall"],
#    "FAKE F1": lr_report["FAKE"]["f1-score"],
#    "REAL Precision": lr_report["REAL"]["precision"],
#    "REAL Recall": lr_report["REAL"]["recall"],
#    "REAL F1": lr_report["REAL"]["f1-score"],
#    "Macro F1": lr_report["macro avg"]["f1-score"],
#    "Errors": (Y_test != y_pred).sum()
#  },
#
#  {
#     "Model": "Naive Bayes",
#     "Accuracy": nb_accuracy,
#     "FAKE Precision": nb_report["FAKE"]["precision"],
#     "FAKE Recall": nb_report["FAKE"]["recall"],
#     "FAKE F1": nb_report["FAKE"]["f1-score"],
#     "REAL Precision": nb_report["REAL"]["precision"],
#     "REAL Recall": nb_report["REAL"]["recall"],
#     "REAL F1": nb_report["REAL"]["f1-score"],
#     "Macro F1": nb_report["macro avg"]["f1-score"],
#     "Errors": (Y_test != nb_y_pred).sum()
#  },
#
#  {
#     "Model": "Linear SVM",
#     "Accuracy": svm_accuracy,
#     "FAKE Precision": svm_report["FAKE"]["precision"],
#     "FAKE Recall": svm_report["FAKE"]["recall"],
#     "FAKE F1": svm_report["FAKE"]["f1-score"],
#     "REAL Precision": svm_report["REAL"]["precision"],
#     "REAL Recall": svm_report["REAL"]["recall"],
#     "REAL F1": svm_report["REAL"]["f1-score"],
#     "Macro F1": svm_report["macro avg"]["f1-score"],
#     "Errors": (Y_test != svm_y_pred).sum()
#    },
#
#    {
#      "Model": "Random Forest",
#      "Accuracy": rf_accuracy,
#      "FAKE Precision": rf_report["FAKE"]["precision"],
#      "FAKE Recall": rf_report["FAKE"]["recall"],
#      "FAKE F1": rf_report["FAKE"]["f1-score"],
#      "REAL Precision": rf_report["REAL"]["precision"],
#      "REAL Recall": rf_report["REAL"]["recall"],
#      "REAL F1": rf_report["REAL"]["f1-score"],
#      "Macro F1": rf_report["macro avg"]["f1-score"],
#      "Errors": (Y_test != rf_y_pred).sum()
#    }
#]
#
#results_df= pd.DataFrame(results)
#
#print("\nModel Comparison:")
#print(results_df)
#



#plt.figure(figsize=(8,5))
#
#plt.bar(
#  results_df["Model"],
#  results_df["Accuracy"]
#
#)
#
#plt.ylabel("Accuracy")
#plt.title("Model Accuracy Comparison")
#plt.xticks(rotation=15)
#
#plt.tight_layout()
#plt.show()

#train_titles= set(X_train.str.split().str[:15].str.join(" "))
#test_titles= set(X_test.str.split().str[:15].str.join(" "))
#
#overlap = train_titles.intersection(test_titles)
#
#print("Number of overlapping title prefixes:", len(overlap))
#
#train_title_set= set(df.loc[X_train.index, "title"])
#test_title_set= set(df.loc[X_test.index, "title"])
#
#title_overlap = train_title_set.intersection(test_title_set)
#
#print("Exact title overlap:", len(title_overlap))
#
#subject_label_counts= pd.crosstab(df["subject"], df["label"])
#print(subject_label_counts)
#
#subject_prediction= df["subject"].map(
#  {
#    "Government News": "FAKE",
#    "News": "FAKE",
#    "US_News": "FAKE",
#    "left-news": "FAKE",
#    "politics": "FAKE",
#    "politicsNews": "REAL",
#    "worldnews": "REAL"
#  }
#)
#subject_accuracy= accuracy_score(df["label"], subject_prediction)
#print("Subject-only accuracy:", subject_accuracy)
#
#
#print("SVM class:",svm_model.classes_)
#
#feature_names = tfidf.get_feature_names_out()
#svm_coefficients= svm_model.coef_[0]
#
#feature_weights = pd.DataFrame({
#  "feature": feature_names,
#  "weight": svm_coefficients
#})
#
#print("Features pushing toward REAL:")
#print(
#  feature_weights.sort_values("weight", ascending=False).head(20)
#  .to_string(index=False)
#)
#
#print("\nFeatures pushing toward FAKE:")
#print(
#  feature_weights.sort_values("weight", ascending=True)
#  .head(20).to_string(index=False)
#  )
#
#testing


#X_train_no_reuters = X_train.str.replace(
#  "reuters","",case=False, regex=False
#)
#X_test_no_reuters = X_test.str.replace(
#  "reuters","",case=False, regex=False
#)
#
#print("Original training text:")
#print(X_train.iloc[0][:200])
#
#print("\nModified training text:")
#print(X_train_no_reuters.iloc[0][:200])
#
#tfidf_no_reuters= TfidfVectorizer()
#
#X_train_tfidf_no_reuters= tfidf_no_reuters.fit_transform(
#  X_train_no_reuters
#)
#
#X_test_tfidf_no_reuters= tfidf_no_reuters.transform(
#  X_test_no_reuters
#)
#
#print("Training shape:", X_train_tfidf_no_reuters.shape)
#print("Testing shape:", X_test_tfidf_no_reuters.shape)
#
#svm_no_reuters = LinearSVC()
#
#svm_no_reuters.fit(
#    X_train_tfidf_no_reuters,
#    Y_train
#)
#
#y_pred_no_reuters = svm_no_reuters.predict(
#  X_test_tfidf_no_reuters
#)
#
#accuracy_no_reuters = accuracy_score(
#  Y_test,
#  y_pred_no_reuters
#)
#
#print("Original SVM accuracy:", svm_accuracy)
#print("SVM accuracy without Reuters:", accuracy_no_reuters)
#
#print(
#  "Accuracy change:",
#  svm_accuracy - accuracy_no_reuters
#)
#
## Only using title instead of content
#
#X_train_title= df.loc[X_train.index, "title"]
#X_test_title= df.loc[X_test.index, "title"]
#print("Training titles:", X_train_title.shape)
#print("Testing titles:", X_test_title.shape)
#
#print("\nExample title:")
#print(X_train_title.iloc[0])
#
#
#print("Training titles:", X_train_title.shape)
#print("Testing titles:", X_test_title.shape)
#
#print("\nExample title:")
#print(X_train_title.iloc[0])
#
#title_tfidf = TfidfVectorizer()
#
#X_train_title_tfidf = title_tfidf.fit_transform(X_train_title)
#X_test_title_tfidf = title_tfidf.transform(X_test_title)
#
#print("Title TF-IDF training shape:", X_train_title_tfidf.shape)
#print("Title TF-IDF testing shape:", X_test_title_tfidf.shape)
#
#title_svm = LinearSVC()
#title_svm.fit(X_train_title_tfidf, Y_train)
#
#title_pred = title_svm.predict(
#  X_test_title_tfidf
#)
#
#title_accuracy= accuracy_score(
#  Y_test, title_pred
#)
#
#print("Title-only accuracy", title_accuracy)
#
#
#

#              only body instead of content

#X_train_body = df.loc[X_train.index, "text"]
#X_test_body = df.loc[X_test.index, "text"]
#
#print("Training bodies:", X_train_body.shape)
#print("Testing bodies:", X_test_body.shape)
#
#print("\nExample article body:")
#print(X_train_body.iloc[0][:500])
#
#body_tfidf = TfidfVectorizer()
#
#X_train_body_tfidf = body_tfidf.fit_transform(X_train_body)
#
#X_test_body_tfidf = body_tfidf.transform(X_test_body)
#
#print("Body TF-IDF training shape:", X_train_body_tfidf.shape)
#print("Body TF-IDF testing shape:", X_test_body_tfidf.shape)
#
#body_svm = LinearSVC()
#body_svm.fit(X_train_body_tfidf, Y_train)
#
#body_pred = body_svm.predict(
#  X_test_body_tfidf
#)
#
#body_accuracy= accuracy_score(
#  Y_test, body_pred
#)
#
#print("Body-only accuracy", body_accuracy)
#
#from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#
#body_svm = LinearSVC()
#body_svm.fit(X_train_body_tfidf, Y_train)
#
#body_pred = body_svm.predict(X_test_body_tfidf)
#
#print("Body-only accuracy:", accuracy_score(Y_test, body_pred))
#
#print("\nClassification Report:")
#print(classification_report(Y_test, body_pred))
#
#print("\nConfusion Matrix:")
#print(confusion_matrix(Y_test, body_pred))
#
X_train_123, X_test_123,  Y_train_123, Y_test_123 = train_test_split(
  X,Y,test_size=0.2, random_state=123, stratify=Y
)
#print("Training size:", X_train_123.shape)
#print("Testing size:", X_test_123.shape)
#
#
tfidf_123= TfidfVectorizer()
X_train_tfidf_123= tfidf_123.fit_transform(X_train_123)
X_test_tfidf_123= tfidf_123.transform(X_test_123)

#print("New TF-IDF training shape:", X_train_tfidf_123.shape)
#print("New TF-IDF testing shape:", X_test_tfidf_123.shape)
#
#svm_model=LinearSVC()
#svm_model.fit(X_train_tfidf_123, Y_train_123)
#
#svm_y_pred_123=svm_model.predict(X_test_tfidf_123)
#print("Number of SVM predictions:", len(svm_y_pred_123))
#print("First 10 predictions of SVM:", svm_y_pred_123[:10])
#print("Actual svm labels:", Y_test_123.iloc[:10].values)
#svm_accuracy_123 =accuracy_score(Y_test_123, svm_y_pred_123)
#
#print("SVM Accuracy:", svm_accuracy_123)
#print("SVM Accuracy percentage:", svm_accuracy_123*100)
#print("\nSVM classification report:")
#print(classification_report(Y_test_123,svm_y_pred_123))
#
## XGB boost
#
#xgb_tfidf = TfidfVectorizer(max_features=20000)
#
#X_train_xgb = xgb_tfidf.fit_transform(X_train)
#X_test_xgb = xgb_tfidf.transform(X_test)
#
#print("XGBoost TF-IDF training shape:", X_train_xgb.shape)
#print("XGBoost TF-IDF testing shape:", X_test_xgb.shape)
#
#Y_train_xgb = Y_train.map({
#    "FAKE": 0,
#    "REAL": 1
#})
#
#print("XGBoost training labels:")
#print(Y_train_xgb.value_counts())
#
#xgb_model=XGBClassifier(
#  n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8,
#  colsample_bytree=0.8, random_state=42, n_jobs=-1,eval_metric="logloss"
#)
#
#xgb_model.fit(X_train_xgb, Y_train_xgb)
#print("XGBoost training completed!")
#
#xgb_pred_numeric= xgb_model.predict(X_test_xgb)
#
#print("First 10 numeric predictions:")
#print(xgb_pred_numeric[:10])
#
#xgb_pred= pd.Series(xgb_pred_numeric).map({
#  0: "FAKE",
#  1: "REAL"
#})
#
#print("\nFirst 10 XGBoost predictions:")
#print(xgb_pred.iloc[:10].values)
#print("\nActual labels:")
#print(Y_test.iloc[:10].values)
#
#xgb_accuracy= accuracy_score(Y_test, xgb_pred)
#print("XGBoost Accuracy:", xgb_accuracy)
#print("XGBoost Accuracy percentage:", xgb_accuracy * 100)
#
#print("\nXGBoost Classification Report:")
#print(classification_report(Y_test, xgb_pred))
#
#print("\nXGBoost Confusion Matrix:")
#print(confusion_matrix(Y_test, xgb_pred))
#
#svm_20k= LinearSVC()
#svm_20k.fit(X_train_xgb, Y_train)
#svm_20k_pred = svm_20k.predict(X_test_xgb)
#
#svm_20k_accuracy = accuracy_score(Y_test, svm_20k_pred)
#
#print("SVM 20k Accuracy:", svm_20k_accuracy)
#print("SVM 20k Accuracy percentage:", svm_20k_accuracy * 100)
#
#print("\nSVM 20k Classification Report:")
#print(classification_report(Y_test, svm_20k_pred))
#
#print("\nSVM 20k Confusion Matrix:")
#print(confusion_matrix(Y_test, svm_20k_pred))
#
xgb_tfidf_123= TfidfVectorizer(max_features=20000)

X_train_xgb_123= xgb_tfidf_123.fit_transform(X_train_123)
X_test_xgb_123= xgb_tfidf_123.transform(X_test_123)

print("XGBoost 123 TF-IDF training shape:", X_train_xgb_123.shape)
print("XGBoost 123 TF-IDF testing shape:", X_test_xgb_123.shape)


xgb_model_123 = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss"
)

Y_train_xgb_123 = Y_train_123.map({
    "FAKE": 0,
    "REAL": 1
})

xgb_model_123.fit(X_train_xgb_123, Y_train_xgb_123)

print("XGBoost 123 training completed!") 

xgb_pred_123_numeric= xgb_model_123.predict(X_test_xgb_123)

xgb_pred_123= pd.Series(xgb_pred_123_numeric).map({
  0: "FAKE",
  1: "REAL"
})

print("First 10 predictions:")
print(xgb_pred_123.head(10).to_list())

print("\nFirst 10 actual labels:")
print(Y_test_123.head(10).to_list())

xgb_accuracy_123 = accuracy_score(
    Y_test_123,
    xgb_pred_123
)

print("XGBoost accuracy on split 123:", xgb_accuracy_123)
print("XGBoost accuracy (%):", xgb_accuracy_123 * 100)

print("\nXGBoost Classification Report:")
print(classification_report(Y_test_123, xgb_pred_123))

print("\nXGBoost Confusion Matrix:")
print(confusion_matrix(Y_test_123, xgb_pred_123))

svm_20k_123 = LinearSVC()

svm_20k_123.fit(
    X_train_xgb_123,
    Y_train_123
)

svm_20k_pred_123 = svm_20k_123.predict(
    X_test_xgb_123
)

svm_20k_accuracy_123 = accuracy_score(
    Y_test_123,
    svm_20k_pred_123
)

print("SVM 20k accuracy on split 123:", svm_20k_accuracy_123)
print("SVM 20k accuracy (%):", svm_20k_accuracy_123 * 100)

print("First 10 SVM predictions:")
print(svm_20k_pred_123[:10])

print("\nFirst 10 XGBoost predictions:")
print(xgb_pred_123.head(10).to_list())

print("\nFirst 10 SVM == XGBoost:")
print(
    svm_20k_pred_123[:10]
    == xgb_pred_123.head(10).to_numpy()
)

svm_predictions_array = svm_20k_pred_123
xgb_predictions_array = xgb_pred_123.to_numpy()

svm_xgb_disagree = svm_predictions_array != xgb_predictions_array

print("Number of disagreements:", svm_xgb_disagree.sum())
print("Number of agreements:", (~svm_xgb_disagree).sum())


disagreement_actual = Y_test_123.to_numpy()[svm_xgb_disagree]
disagreement_svm= svm_predictions_array[svm_xgb_disagree]
disagreement_xgb= xgb_predictions_array[svm_xgb_disagree]

svm_correct= disagreement_svm == disagreement_actual
xgb_correct = disagreement_xgb == disagreement_actual

print("Disagreement cases:", len(disagreement_actual))
print("SVM correct:", svm_correct.sum())
print("XGBoost correct:", xgb_correct.sum())

ensemble_pred_123 = xgb_predictions_array.copy()

ensemble_pred_123[svm_xgb_disagree] = svm_predictions_array[svm_xgb_disagree]

ensemble_accuracy_123 = accuracy_score(
    Y_test_123,
    ensemble_pred_123
)

print("XGBoost accuracy:", xgb_accuracy_123)
print("SVM + XGBoost ensemble accuracy:", ensemble_accuracy_123)
print("Ensemble accuracy (%):", ensemble_accuracy_123 * 100)

xgb_report_123 = classification_report(
    Y_test_123,
    xgb_pred_123,
    output_dict=True
)

print("XGBoost 123 metrics:")
print("FAKE Precision:", xgb_report_123["FAKE"]["precision"])
print("FAKE Recall:", xgb_report_123["FAKE"]["recall"])
print("FAKE F1:", xgb_report_123["FAKE"]["f1-score"])
print("REAL Precision:", xgb_report_123["REAL"]["precision"])
print("REAL Recall:", xgb_report_123["REAL"]["recall"])
print("REAL F1:", xgb_report_123["REAL"]["f1-score"])
print("Macro F1:", xgb_report_123["macro avg"]["f1-score"])

xgb_errors_123 = (Y_test_123.to_numpy() != xgb_pred_123.to_numpy()).sum()

print("Total errors:", xgb_errors_123)

print("Total articles:", len(X))
print("Labels:")
print(Y.value_counts())

final_tfidf = TfidfVectorizer(max_features=20000)

X_final_tfidf = final_tfidf.fit_transform(X)

print("Final TF-IDF shape:", X_final_tfidf.shape)

Y_final_xgb = Y.map({"FAKE": 0, "REAL":1})
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
  X_final_tfidf, Y_final_xgb
)
print("Final XGBoost model trained successfully.")

import joblib

joblib.dump(final_xgb_model, "final_xgb_model.pkl")
joblib.dump(final_tfidf, "final_tfidf.pkl")

print("Model and TF-IDF vectorizer saved successfully.")

new_article = """
The government announced a new education policy today,
with officials saying the changes will be introduced next year.
"""

def predict_news(article):

    article_tfidf = final_tfidf.transform([article])

    prediction = final_xgb_model.predict(article_tfidf)

    if prediction[0] == 0:
        return "FAKE"
    else:
        return "REAL"

result = predict_news(new_article)

print(result)