

import unittest
import mlflow
import dagshub
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle


class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # -----------------------------
        # CI Guard
        # -----------------------------
        if os.getenv("CI") == "true":
            print("Running inside GitHub Actions")

        # -----------------------------
        # Auth
        # -----------------------------
        dagshub_token = os.getenv("DAGSHUB_TOKEN")

        if not dagshub_token:
            raise EnvironmentError(
                "DAGSHUB_TOKEN environment variable is not set"
            )
        os.environ["DAGSHUB_USER_TOKEN"] = dagshub_token

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        # -----------------------------
        # MLflow + DagsHub Setup
        # -----------------------------
        repo_owner = "sreesh49"
        repo_name = "YT-Capstone-Project"

        mlflow_tracking_uri = (
            f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
        )

        mlflow.set_tracking_uri(mlflow_tracking_uri)

        dagshub.init(
            repo_owner=repo_owner,
            repo_name=repo_name,
            mlflow=True,
            
        )

        # -----------------------------
        # Load latest registered model
        # -----------------------------
        cls.new_model_name = "my_model"

        cls.new_model_version = cls.get_latest_model_version(
            cls.new_model_name
        )

        if cls.new_model_version is None:
            raise ValueError("No model version found in MLflow Registry")

        cls.new_model_uri = (
            f"models:/{cls.new_model_name}/{cls.new_model_version}"
        )

        print(f"Loading model from: {cls.new_model_uri}")

        cls.new_model = mlflow.pyfunc.load_model(
            cls.new_model_uri
        )

        # -----------------------------
        # Load vectorizer
        # -----------------------------
        with open("models/vectorizer.pkl", "rb") as f:
            cls.vectorizer = pickle.load(f)

        # -----------------------------
        # Load test dataset
        # -----------------------------
        cls.holdout_data = pd.read_csv(
            "data/processed/test_bow.csv"
        )

    # -------------------------------------------------

    @staticmethod
    def get_latest_model_version(model_name, stage="Staging"):

        client = mlflow.MlflowClient()

        latest_version = client.get_latest_versions(
            model_name,
            stages=[stage]
        )

        if not latest_version:
            latest_version = client.get_latest_versions(
                model_name,
                stages=["None"]
            )

        return latest_version[0].version if latest_version else None

    # -------------------------------------------------

    def test_model_loaded_properly(self):

        self.assertIsNotNone(self.new_model)

    # -------------------------------------------------

    def test_model_signature(self):

        input_text = "hi how are you"

        input_data = self.vectorizer.transform([input_text])

        input_df = pd.DataFrame(
            input_data.toarray(),
            columns=[
                str(i)
                for i in range(input_data.shape[1])
            ]
        )

        prediction = self.new_model.predict(input_df)

        # input shape check
        self.assertEqual(
            input_df.shape[1],
            len(self.vectorizer.get_feature_names_out())
        )

        # output shape check
        self.assertEqual(
            len(prediction),
            input_df.shape[0]
        )

        self.assertEqual(
            len(prediction.shape),
            1
        )

    # -------------------------------------------------

    def test_model_performance(self):

        X_holdout = self.holdout_data.iloc[:, 0:-1]
        y_holdout = self.holdout_data.iloc[:, -1]

        y_pred_new = self.new_model.predict(X_holdout)

        accuracy_new = accuracy_score(
            y_holdout,
            y_pred_new
        )

        precision_new = precision_score(
            y_holdout,
            y_pred_new
        )

        recall_new = recall_score(
            y_holdout,
            y_pred_new
        )

        f1_new = f1_score(
            y_holdout,
            y_pred_new
        )

        # Thresholds
        expected_accuracy = 0.40
        expected_precision = 0.40
        expected_recall = 0.40
        expected_f1 = 0.40

        self.assertGreaterEqual(
            accuracy_new,
            expected_accuracy
        )

        self.assertGreaterEqual(
            precision_new,
            expected_precision
        )

        self.assertGreaterEqual(
            recall_new,
            expected_recall
        )

        self.assertGreaterEqual(
            f1_new,
            expected_f1
        )


if __name__ == "__main__":
    unittest.main()