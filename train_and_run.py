import nltk
import random
import pandas as pd
from nltk.corpus import movie_reviews
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Download dataset
nltk.download('movie_reviews')

# Load and shuffle data
docs = [(list(movie_reviews.words(fileid)), category)
        for category in movie_reviews.categories()
        for fileid in movie_reviews.fileids(category)]
random.shuffle(docs)

# Convert to text and labels
texts = [" ".join(words) for words, label in docs]
labels = [1 if label == 'pos' else 0 for _, label in docs]
df = pd.DataFrame({'text': texts, 'label': labels})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Evaluate model
print("\n=== Evaluation ===")
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# Save model + vectorizer
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("✅ Model and vectorizer saved.")

# Interactive prediction
print("\n=== Try it out ===")
while True:
    text = input("Enter a sentence (or 'q' to quit): ")
    if text.lower() == 'q':
        break
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    sentiment = "Positive" if pred == 1 else "Negative"
    print(f"Predicted sentiment: {sentiment}\n")
