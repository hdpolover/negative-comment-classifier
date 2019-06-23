
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

import time
start = time.time()

df = pd.read_csv("data\comment_data_RAW.csv")

df_data = df[["comment_text", "toxic", "severe_toxic", "obscene",
       "threat", "insult", "identity_hate"]]

df_x = df_data["comment_text"]
df_y = df_data[["toxic", "severe_toxic", "obscene",
       "threat", "insult", "identity_hate"]]

vect = TfidfVectorizer(stop_words="english", sublinear_tf=True)
x = vect.fit_transform(df_x)

#split data
x_train, x_test, y_train, y_test = train_test_split(x, df_y, test_size=0.2, random_state=42, shuffle=True)

clf = LinearSVC()

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
    clf.fit(x_train, y_train[category])
    prediction = clf.predict(vector)
    #add each category prediction to the list
    new_data.append(prediction)
    print('Prediction:', prediction)
    # accuracy = clf.predict(x_test)      
    # print('Prediction accuracy is {}'.format(accuracy_score(y_test[category], accuracy)))

#print list for the new data
print(new_data)

end = time.time()
print("time needed:", end - start)