import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Load data from CSV file
data = pd.read_csv(r'C:\Users\shashiraj.walsetwar\Desktop\Solo-Hackathon\Solo Hackathon\Database\full_data.csv', encoding='unicode_escape')

# Define TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Vectorize questions
question_vectors = vectorizer.fit_transform(data['question'])

# Train the model
model = cosine_similarity(question_vectors)

# Save the model
joblib.dump(model, 'cosineSimilarity_chatbot_model_04062023.joblib')

# Load the model
model = joblib.load('cosineSimilarity_chatbot_model.joblib')

# Define function to generate responses
def generate_response(query):
    query_vector = vectorizer.transform([query])
    scores = list(enumerate(cosine_similarity(query_vector, question_vectors)[0]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    print(f'sorted_scores[0][1]: {sorted_scores[0][1]}')
    if sorted_scores[0][1] == 0:
        return "I'm sorry, I don't understand."
    else:
        return data['answer'][sorted_scores[0][0]]

# Load the model
model = joblib.load('cosineSimilarity_chatbot_model.joblib')

while True:
    # Define a user query
    query = input('You: ')

    # Generate a response
    response = generate_response(query)

    # Print the response
    print(response)
