import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from core.database import load_data

def train_model():
    data = load_data()
    if not data:
        print("No data to train on.")
        return None

    df = pd.DataFrame(data)
    X = df["command"]
    y = df["response"]

    model = Pipeline([('vect', CountVectorizer()), ('clf', MultinomialNB())])
    model.fit(X, y)
    
    return model

def predict_response(model, command):
    if model:
        return model.predict([command])[0]
    else:
        return "I don't know the answer to that yet."