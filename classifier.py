import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

df = pd.read_csv("data\comment_data_RAW.csv")

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

# make the classifier
clf = LinearSVC()

categories = ['toxic', 'severe_toxic', 'obscene',
              'threat', 'insult', 'identity_hate']

# list for text and predictions
pr = []  # prediction result
pa = []  # prediction accuracy


def predAll(text):
    # input text
    data = [text]
    vector = vect.transform(data).toarray()

    for category in categories:
        # fit train data
        clf.fit(x_train, y_train[category])
        prediction = clf.predict(vector)
        # add each category prediction to the list
        pr.append(prediction)
        # accuracy
        accuracy = clf.predict(x_test)
        accuracy_result = accuracy_score(y_test[category], accuracy)
        # add each category accuracy to the list
        pa.append(accuracy_result)

    return pr[0], pr[1], pr[2], pr[3], pr[4], pr[5]
    #   return pr[0], pr[1], pr[2], pr[3], pr[4], pr[5], pa[0], pa[1], pa[2],
    # pa[3], pa[4], pa[5]


def pred(text, choice):
        # input text
    data = [text]
    vector = vect.transform(data).toarray()

    # fit train data
    clf.fit(x_train, y_train[categories[choice]])
    prediction = clf.predict(vector)
    # accuracy
    accuracy = clf.predict(x_test)
    accuracy_result = accuracy_score(
        y_test[categories[choice]], accuracy)

    return prediction, accuracy_result
