import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
text = input("Write  news to check is it False or True: ")
print("The News is:",text)
pkl_filename = "testmodel.pkl"
try:
    with open(pkl_filename, 'rb') as file:
        model = pickle.load(file)
        tf1 = pickle.load(open("tfidf.pkl", 'rb'))
    print("Model Runned Successfully.")
except:
    print("Model not found")
    

tf1_new = TfidfVectorizer(analyzer='word', stop_words = "english", vocabulary = tf1.vocabulary_)

X_temp = tf1_new.fit_transform([text])
X_temp.toarray()
predict = model.predict(X_temp)
print(predict)
