import unittest
import pandas as pd
from src.data_preprocessing import load_and_clean_data, split_data
from src.model_training import get_models, train_models
from src.evaluation import evaluate_model

class TestFetalHealthPipeline(unittest.TestCase):

    def setUp(self):
        # Dummy dataset for testing
        data = {
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1],
            'fetal_health': [1, 2, 1, 3, 2]
        }
        df = pd.DataFrame(data)
        self.df = df
        self.X_train, self.X_test, self.y_train, self.y_test = split_data(df)

    def test_model_training_and_evaluation(self):
        models = get_models()
        trained_models = train_models(models, self.X_train, self.y_train)
        for name, model in trained_models.items():
            accuracy, _ = evaluate_model(model, self.X_test, self.y_test)
            self.assertGreaterEqual(accuracy, 0.0)

if __name__ == "__main__":
    unittest.main()
