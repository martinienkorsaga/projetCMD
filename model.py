
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def create_models():
    """Crée et retourne les modèles avec leurs hyperparamètres"""
    models = {
        'RandomForest': {
            'model': RandomForestClassifier(),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            }
        },
        'LogisticRegression': {
            'model': LogisticRegression(),
            'params': {
                'C': [0.1, 1, 10],
                'max_iter': [1000]
            }
        },
        'SVM': {
            'model': SVC(probability=True),
            'params': {
                'C': [1, 10],
                'kernel': ['rbf', 'linear']
            }
        }
    }
    return models

def train_model(model, X_train, y_train, params=None):
    """Entraîne un modèle avec recherche des meilleurs hyperparamètres"""
    if params:
        grid_search = GridSearchCV(model, params, cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_
    else:
        model.fit(X_train, y_train)
        return model

def predict(model, X):
    """Fait des prédictions avec le modèle"""
    return model.predict(X)

def predict_proba(model, X):
    """Retourne les probabilités des prédictions"""
    return model.predict_proba(X)

def evaluate_model(model, X_test, y_test):
    """Évalue les performances du modèle"""
    predictions = predict(model, X_test)
    return accuracy_score(y_test, predictions)
