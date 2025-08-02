from src.data_preprocessing import load_and_clean_data, split_data
from src.model_training import get_models, train_models
from src.evaluation import evaluate_all_models

def main():
    # 1. Load data
    df = load_and_clean_data("fetal_health.csv")
    print(f"Dataset loaded with {len(df)} rows")

    # 2. Split data
    X_train, X_test, y_train, y_test = split_data(df)

    # 3. Get and train models
    models = get_models()
    trained_models = train_models(models, X_train, y_train)

    # 4. Evaluate
    results = evaluate_all_models(trained_models, X_test, y_test)

    # 5. Display results
    for model_name, metrics in results.items():
        print(f"\nModel: {model_name}")
        print(f"Accuracy: {metrics['accuracy']:.2f}")
        print(metrics['report'])

if __name__ == "__main__":
    main()
