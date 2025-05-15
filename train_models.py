import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import os

def train_and_save_models():
    try:
        # Load the data
        print("Loading data...")
        data = pd.read_csv('main_data.csv')
        
        # Initialize and fit the TfidfVectorizer
        print("Training TfidfVectorizer...")
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(data['comb'])
        
        # Save the vectorizer
        print("Saving vectorizer...")
        with open('tranform.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
        
        # Train a simple classifier (you can modify this based on your needs)
        print("Training classifier...")
        # For demonstration, we'll use a simple binary classification
        # You should replace this with your actual training data and labels
        y = np.random.randint(0, 2, size=len(data))  # Replace with actual labels
        clf = MultinomialNB()
        clf.fit(X, y)
        
        # Save the classifier
        print("Saving classifier...")
        with open('nlp_model.pkl', 'wb') as f:
            pickle.dump(clf, f)
            
        print("Models trained and saved successfully!")
        
    except Exception as e:
        print(f"Error training models: {str(e)}")
        raise

if __name__ == "__main__":
    train_and_save_models() 