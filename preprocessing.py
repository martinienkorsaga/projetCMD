```python
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import re

def clean_text(text):
    """Nettoie le texte des tweets"""
    # Conversion en minuscules
    text = str(text).lower()
    
    # Suppression des mentions, hashtags, URLs et caractères spéciaux
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    
    # Suppression des espaces multiples
    text = ' '.join(text.split())
    
    return text

def tokenize_text(text):
    """Tokenize le texte et retire les stop words"""
    # Téléchargement des ressources NLTK nécessaires
    nltk.download('punkt')
    nltk.download('stopwords')
    
    # Tokenisation
    tokens = word_tokenize(text)
    
    # Suppression des stop words
    stop_words = set(stopwords.words('french'))
    tokens = [token for token in tokens if token not in stop_words]
    
    return ' '.join(tokens)

def prepare_data(df):
    """Prépare les données pour l'entraînement"""
    # Nettoyage et tokenisation
    X = df['message'].apply(clean_text).apply(tokenize_text)
    y = df['label']
    
    # Vectorisation TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X_vectorized = vectorizer.fit_transform(X)
    
    return X_vectorized, y, vectorizer

def vectorize_new_data(text, vectorizer):
    """Vectorise de nouvelles données avec le même vectorizer"""
    cleaned_text = clean_text(text)
    tokenized_text = tokenize_text(cleaned_text)
    return vectorizer.transform([tokenized_text])
```