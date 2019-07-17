from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

df = pd.read_csv(
    "data\comment_data_RAW.csv")

df_data = df[["comment_text", "toxic", "severe_toxic", "obscene",
              "threat", "insult", "identity_hate"]]

df_x = df_data["comment_text"]
df_y = df_data[["toxic", "severe_toxic", "obscene",
                "threat", "insult", "identity_hate"]]

vect = TfidfVectorizer(stop_words="english", sublinear_tf=True)
x = vect.fit_transform(df_x)

y = df_y
    
categories = ['identity_hate', 'insult', 'obscene',
              'threat', 'toxic', 'severe_toxic']

clf = LinearSVC()

n_folds = 10
skf = KFold(n_splits = n_folds)

score = 0.0

print("==Average Kfold Score for n_folds = {}==".format(n_folds))
for category in categories:
    for train_index, test_index in skf.split(x):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[category][train_index], y[category][test_index]
        clf.fit(X_train,y_train)
        pred = clf.predict(X_test)
        score = score + roc_auc_score(y_test,pred)

    avg_score = score/n_folds
    print("- {} \t: {}".format(category, avg_score))
    score = 0.0

