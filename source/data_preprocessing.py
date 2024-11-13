import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re

nltk.download('stopwords')
from nltk.corpus import stopwords

def load_data():
    df = pd.read_csv("data/IMDB_Dataset.csv")
    return df

def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)

def vectorize_data(df):
    vectorizer = TfidfVectorizer(max_features=5000)
    x = vectorizer.fit_transform(df['review'].apply(preprocess_text))
    y = df['sentiment']
    return train_test_split(x, y, test_size=0.2, random_state=42)