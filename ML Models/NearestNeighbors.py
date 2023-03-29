# Importing required libraries
import pandas as pd
import numpy as np
import re
import string

# Importing dataset
df = pd.read_csv('Database\InputData.csv')

# Preprocessing the dataset
def preprocess(text):
    # Removing punctuations
    text = str(text).translate(str.maketrans("", "", string.punctuation))
    
    # Converting text to lowercase
    text = str(text).lower()
    
    # Removing extra whitespaces
    text = re.sub('\s+', ' ', text).strip()
    
    return text

df['question'] = df['question'].apply(preprocess)
df['answer'] = df['answer'].apply(preprocess)

# Vectorizing the dataset
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['question'])

# Training a machine learning model
from sklearn.neighbors import NearestNeighbors
model = NearestNeighbors(n_neighbors=1)
model.fit(X)

# Testing the model
def get_answer(question):
    question = preprocess(question)
    question_vector = vectorizer.transform([question])
    _, index = model.kneighbors(question_vector)
    return df['answer'][index[0][0]]

# Example usage

sample_questions = ['Who is Prasad Khandat?', 'what is the sikka.ai product for revenue cycle management industry?', 'what domains is Optimizer available for?','what does sikka.ai do?','who is the CFO of sikka.ai?']

for question in sample_questions:
    # Preprocess the question
    answer = get_answer(question)
    print(f"Q: {question}\nA: {answer}\n")

# question = "What is machine learning?"
# answer = get_answer(question)
# print(answer)
