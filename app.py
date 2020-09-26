#In this file, we will use the flask web framework to handle the POST requests that we will get from the request.py and from HTML file

#import packages
import os
import numpy as np
import flask
from flask import Flask, request, jsonify,  render_template
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

# load the model from disk
filename = 'tfidf.pkl'
clf = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('testmodel.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

	if request.method == 'POST':
		news = request.form['news']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)
