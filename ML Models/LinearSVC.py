import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import joblib

try:
    df = pd.read_csv('Database\InputData.csv')
    print(f'Database Imported Successfully')
except:
    print(f'Database Import Failed. Exiting Program')
    exit()

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    # Tokenize the text
    words = word_tokenize(str(text).lower())
    # Remove stop words and punctuation
    # words = [stemmer.stem(word) for word in words if word.isalpha() and word not in stop_words]
    # # Perform lemmatization
    # words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)
    
try:
    df['Question'] = df['question'].apply(preprocess)
    df['Answer'] = df['answer'].apply(preprocess)
    print('Preprocessing on Question/ Answer Success')
except:
    print('Preprocessing on Question/ Answer Failed')
    exit()

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Question'])
y = df['Answer']

best_score = 0
best_model = None

# Train the model multiple times with different random seeds
for i in range(100):
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42) 
    # Initialize the model with a random seed
    clf = LinearSVC(random_state=i)

    # Train the model on the training data
    clf.fit(X_train, y_train)

    # Evaluate the model on the validation data
    score = clf.score(X_val, y_val)

    # If this model has a higher accuracy, update the best model
    print(f'Score: {score}')
    if score > best_score:
        best_score = score
        best_model = clf


print(f"Best accuracy: {best_score}")
joblib.dump(best_model, 'best_model.joblib')

sample_questions = ['Who is Prasad Khandat?', 'what is the sikka.ai product for insurance industry?', 'what domains is Optimizer available for?','what does sikka.ai do?','who is the CFO of sikka.ai?']

for question in sample_questions:
    # Preprocess the question
    question = preprocess(question)
    # Vectorize the question
    question_vec = vectorizer.transform([question])
    # Predict the answer
    answer = clf.predict(question_vec)[0]
    print(f"Q: {question}\nA: {answer}\n")




