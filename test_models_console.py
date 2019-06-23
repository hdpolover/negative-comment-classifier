from joblib import load
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import time
start = time.time()

df = pd.read_csv(
    "data\comment_data_RAW.csv")

df_x = df["comment_text"]

vect = TfidfVectorizer(stop_words="english", sublinear_tf=True)
x = vect.fit_transform(df_x)

#input text
comment = "fuck you"
data = [comment]
vector = vect.transform(data).toarray()
print("predict text:", data)

#list for text and predictions
new_data = [comment]

categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
for category in categories:
    print('... Processing {}'.format(category))
    file_name = "model_{}.pkl".format(category)
    clf = load(file_name)
    prediction = clf.predict(vector)
    #add each category prediction to the list
    new_data.append(prediction)
    print('Prediction:', prediction)

end = time.time()
print("time needed:", end - start)