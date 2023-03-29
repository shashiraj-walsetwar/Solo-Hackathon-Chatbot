import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import pickle

# Load the Q&A dataset
data = pd.read_csv('Database\InputData.csv', encoding= 'unicode_escape')

# Extract features using TF-IDF
tfidf = TfidfVectorizer()
features = tfidf.fit_transform(data['question'])

# Select a machine learning model (e.g., SVM)
model = LinearSVC()

# Train the model
model.fit(features, data['answer'])

pickle.dump(model, open("LSVC_Version_3.0_model", 'wb'))

# Build the chatbot
def chatbot(input_text):
    features = tfidf.transform([input_text])
    output_text = model.predict(features)[0]
    return output_text

while True:
    answer = input("You: ")
    print(f'Bot: {chatbot(answer)}')
