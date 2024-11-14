# projetCMD

# Détection de Tweets Suspects

## Description
Système de détection automatique de tweets suspects utilisant le machine learning. Implémente trois modèles : RandomForest, SVM et Régression Logistique.

## Prérequis
- Python 3.8+
- pip

## Installation
```bash
git clone https://github.com/votre-username/tweet-detection.git
cd tweet-detection
pip install -r requirements.txt
```

## Structure du Projet
```
tweet-detection/
├── notebooks/
│   └── tweet_detection.ipynb
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   └── evaluation.py
└── data/
    └── tweets_suspect.csv
```

## Utilisation

1. Préparation des données :
```python
from src.preprocessing import prepare_data
X_vectorized, y = prepare_data(df)
```

2. Entraînement :
```python
from src.models import train_models
models = train_models(X_train, y_train)
```

3. Évaluation :
```python
from src.evaluation import evaluate_models
results = evaluate_models(models, X_test, y_test)
```

## Execution via Notebook
```bash
jupyter notebook notebooks/tweet_detection.ipynb
```
