def pred(text, choice):
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

   #split data
   x_train, x_test, y_train, y_test = train_test_split(x, df_y, test_size=0.2, random_state=42, shuffle=True)

   clf = LinearSVC()

   #input text
   data = [text]
   vector = vect.transform(data).toarray()

   #list for text and predictions
   prediction_result = []
   prediction_accuracy = []

   categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

   if choice == "6":
      for category in categories:
         #fit train data
         clf.fit(x_train, y_train[category])
         prediction = clf.predict(vector)
         #add each category prediction to the list
         prediction_result.append(prediction)
         #accuracy
         accuracy = clf.predict(x_test)      
         accuracy_result = accuracy_score(y_test[category], accuracy)
         #add each category accuracy to the list
         prediction_accuracy.append(accuracy_result)

      return prediction_result[0], prediction_result[1], prediction_result[2],
      prediction_result[3], prediction_result[4], prediction_result[5],
      prediction_accuracy[0], prediction_accuracy[1], prediction_accuracy[2],
      prediction_accuracy[3], prediction_accuracy[4], prediction_accuracy[5]
   else:
      #fit train data
      c = int(choice)
      clf.fit(x_train, y_train[categories[c]])
      prediction = clf.predict(vector)
      #accuracy
      accuracy = clf.predict(x_test)      
      accuracy_result = accuracy_score(y_test[categories[c]], accuracy)

      return prediction, accuracy_result
   


