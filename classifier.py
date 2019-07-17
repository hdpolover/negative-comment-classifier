import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import load

df = pd.read_csv("data\comment_data_RAW.csv")

df_x = df["comment_text"]

vect = TfidfVectorizer(stop_words="english", sublinear_tf=True)
x = vect.fit_transform(df_x)

categories = ['identity_hate', 'insult', 'obscene',
              'threat', 'toxic', 'severe_toxic']

# list for text and predictions
pr = []  # prediction result

def predAll(text):
    # input text
    data = [text]
    vector = vect.transform(data).toarray()

    for category in categories:
        # fit train data
        file_name = "model_{}.pkl".format(category)
        clf = load(file_name)
        prediction = clf.predict(vector)
        # add each category prediction to the list
        pr.append(prediction)

    # create a new list
    retPr = []
    # copy list to a new one
    retPr = pr.copy()
    # clear pr elements
    pr.clear()

    return retPr[0], retPr[1], retPr[2], retPr[3], retPr[4], retPr[5]


def pred(text, choice):
    # input text
    data = [text]
    vector = vect.transform(data).toarray()

    # fit train data
    file_name = "model_{}.pkl".format(categories[choice])
    clf = load(file_name)
    prediction = clf.predict(vector)

    return prediction
