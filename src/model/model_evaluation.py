import os
import json
import mlflow
import mlflow.sklearn
import dagshub
import numpy as np
import pandas as pd
import pickle

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from src.logger import logging


# =========================
# MLflow Setup (DVC + CI)
# =========================
def setup_mlflow():
    import os
    import mlflow
    import dagshub

    token = os.getenv("DAGSHUB_TOKEN")

    if not token:
        raise EnvironmentError("DAGSHUB_TOKEN not set")

    # 🔥 IMPORTANT: correct DagsHub MLflow auth
    os.environ["MLFLOW_TRACKING_USERNAME"] = "token"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    mlflow.set_tracking_uri(
        "https://dagshub.com/sreesh49/YT-Capstone-Project.mlflow"
    )

    dagshub.init(
        repo_owner="sreesh49",
        repo_name="YT-Capstone-Project",
        mlflow=True
    )

    mlflow.set_experiment("my-dvc-pipeline")


# =========================
# Utils
# =========================
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_data(path):
    return pd.read_csv(path)


def evaluate(model, X, y):

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "auc": roc_auc_score(y, y_proba)
    }


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


# =========================
# Main
# =========================
def main():

    setup_mlflow()

    with mlflow.start_run() as run:

        model = load_model("./models/model.pkl")
        df = load_data("./data/processed/test_bow.csv")

        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        metrics = evaluate(model, X, y)

        # save locally (DVC artifact)
        save_json(metrics, "reports/metrics.json")

        # log MLflow metrics
        mlflow.log_metrics(metrics)

        # log params (safe check)
        if hasattr(model, "get_params"):
            mlflow.log_params(model.get_params())

        # log model artifact
        mlflow.sklearn.log_model(model, "model")

        # save run info (IMPORTANT for registration step)
        save_json(
            {
                "run_id": run.info.run_id,
                "model_path": "model"
            },
            "reports/experiment_info.json"
        )

        mlflow.log_artifact("reports/metrics.json")

        logging.info("Model evaluation completed")


if __name__ == "__main__":
    main()