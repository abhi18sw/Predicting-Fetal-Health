from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

def get_models():
    """
    Initialize models for training.
    """
    models = {
        "LogisticRegression": LogisticRegression(solver='liblinear', random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=4, criterion='gini', random_state=42),
        "DecisionTree": DecisionTreeClassifier(max_depth=2, criterion='gini', random_state=42)
    }
    return models

def train_models(models, X_train, y_train):
    """
    Train multiple models and return trained instances.
    """
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    return trained_models
