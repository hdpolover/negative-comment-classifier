from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from joblib import dump
import time
start = time.time()


df = pd.read_csv(
    "data\comment_data_RAW.csv")

df_data = df[["comment_text", "toxic", "severe_toxic", "obscene",
              "threat", "insult", "identity_hate"]]

df_x = df_data["comment_text"]
df_y = df_data[["toxic", "severe_toxic", "obscene",
                "threat", "insult", "identity_hate"]]

vect = TfidfVectorizer(stop_words="english", sublinear_tf=True)
x = vect.fit_transform(df_x)

# split data
x_train, x_test, y_train, y_test = train_test_split(
    x, df_y, test_size=0.2, random_state=42, shuffle=True)

clf = LinearSVC()

categories = ['identity_hate', 'insult', 'obscene',
              'threat', 'toxic', 'severe_toxic']
              
for category in categories:
    clf.fit(x_train, y_train[category])
    file_name = "model_{}.pkl".format(category)
    dump(clf, file_name)

end = time.time()
print("time needed:", end - start)
