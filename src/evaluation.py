import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a single model and return accuracy and detailed metrics.
    """
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)
    return accuracy, pd.DataFrame(report).transpose()

def evaluate_all_models(models, X_test, y_test):
    """
    Evaluate all trained models and return results.
    """
    results = {}
    for name, model in models.items():
        acc, report = evaluate_model(model, X_test, y_test)
        results[name] = {"accuracy": acc, "report": report}
    return results
