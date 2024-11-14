from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confusion_matrix(y_true, y_pred, model_name):
   """Affiche la matrice de confusion"""
   cm = confusion_matrix(y_true, y_pred)
   plt.figure(figsize=(8, 6))
   sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
   plt.title(f'Matrice de Confusion - {model_name}')
   plt.ylabel('Vrai label')
   plt.xlabel('Prédiction')
   plt.show()

def plot_roc_curve(y_true, y_pred_proba, model_name):
   """Affiche la courbe ROC"""
   fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
   roc_auc = auc(fpr, tpr)
   
   plt.figure(figsize=(8, 6))
   plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
   plt.plot([0, 1], [0, 1], 'k--')
   plt.xlim([0.0, 1.0])
   plt.ylim([0.0, 1.05])
   plt.xlabel('Taux de faux positifs')
   plt.ylabel('Taux de vrais positifs')
   plt.title(f'Courbe ROC - {model_name}')
   plt.legend(loc="lower right")
   plt.show()

def evaluate_model_performance(model, X_test, y_test, model_name):
   """Évalue les performances complètes du modèle"""
   # Prédictions
   y_pred = model.predict(X_test)
   y_pred_proba = model.predict_proba(X_test)
   
   # Affichage des métriques
   print(f"\nRésultats pour {model_name}:")
   print("\nRapport de classification:")
   print(classification_report(y_test, y_pred))
   
   # Visualisations
   plot_confusion_matrix(y_test, y_pred, model_name)
   plot_roc_curve(y_test, y_pred_proba, model_name)
   
   return {
       'predictions': y_pred,
       'probabilities': y_pred_proba,
       'classification_report': classification_report(y_test, y_pred)
   }

def compare_models(models_results):
   """Compare les performances des différents modèles"""
   for model_name, results in models_results.items():
       print(f"\nModèle: {model_name}")
       print(results['classification_report'])