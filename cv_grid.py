from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np

def cross_validate_models(X, y, cv_splits=5):
    results = {}
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    nb_scores = cross_val_score(GaussianNB(), X, y, cv=skf, scoring='accuracy')
    results['Naive Bayes'] = nb_scores

    dt_scores = cross_val_score(DecisionTreeClassifier(random_state=42), X, y,
                                cv=skf, scoring='accuracy')
    results['Decision Tree'] = dt_scores

    # For MLP normalize inside function
    X_norm = X / 255.0
    mlp_scores = cross_val_score(
        MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, early_stopping=True, random_state=42),
        X_norm, y, cv=skf, scoring='accuracy'
    )
    results['MLP'] = mlp_scores

    return {k: (np.mean(v), np.std(v)) for k, v in results.items()}
