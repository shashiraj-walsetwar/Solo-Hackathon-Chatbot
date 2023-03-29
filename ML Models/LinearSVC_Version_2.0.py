import csv
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import pickle
from sklearn.metrics import accuracy_score


dataset = []

with open('Database\InputData.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip header row
    for row in reader:
        dataset.append({'question': row[0], 'answer': row[1]})

print(f'Got Dataset')


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = str(text).lower()
    tokens = word_tokenize(str(text))
    tokens = [t for t in tokens if t.isalnum() and not t in stop_words]
    # tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)



for data in dataset:
    data['question'] = preprocess_text(data['question'])
    data['answer'] = preprocess_text(data['answer'])

print(f'Preprocessing Complete')

X = [data['question'] for data in dataset]
y = [data['answer'] for data in dataset]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('svc', LinearSVC())
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

filename = 'linear_svc_model_LinearSVC_Version_2.0.sav'
pickle.dump(model, open(filename, 'wb'))

while True:
    question = input("You: ")
    question = preprocess_text(question)
    answer = model.predict([question])[0]
    print(f"Chatbot: {answer}")

