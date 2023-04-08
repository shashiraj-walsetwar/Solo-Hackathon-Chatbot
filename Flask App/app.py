from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import people_also_ask

app = Flask(__name__)

class chatbot:
    def __init__(self):
        # Load data from CSV file
        self.data = pd.read_csv(r'full_data.csv', encoding='unicode_escape')

        # Define TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer()

        # Vectorize questions
        self.question_vectors = self.vectorizer.fit_transform(self.data['question'])

        # loading model
        self.model = joblib.load('cosineSimilarity_chatbot_model.joblib')

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get-response", methods=['POST'])
def get_bot_response():
    user_message = request.form['msg']

    query_vector = chatbot_object.vectorizer.transform([user_message])
    scores = list(enumerate(cosine_similarity(query_vector, chatbot_object.question_vectors)[0]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    response = chatbot_object.data['answer'][sorted_scores[0][0]]
    if sorted_scores[0][1] < 0.50:
        if people_also_ask.get_answer(user_message)['has_answer'] == True and len(people_also_ask.get_answer(user_message)['response']) > 0:
            response_from_google = people_also_ask.get_answer(user_message)['response']
            return response_from_google
        else:
            return 'Sorry, I do not have answer for this question. Could you be more Specific?'
    else:
        print(f'Found Similarity: {sorted_scores[0][1]}')
        return response
    
if __name__ == "__main__":
    chatbot_object = chatbot()
    app.run(debug=True)
