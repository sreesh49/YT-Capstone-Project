import sys
import os

# Add project root to Python path - MUST BE BEFORE src imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import mlflow
import dagshub
from mlflow.tracking import MlflowClient
from src.logger import logging
import warnings

warnings.filterwarnings("ignore")

# -----------------------------
# Auth (IMPORTANT)
# -----------------------------
dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN is not set")

# -----------------------------
# DagsHub MLflow setup
# -----------------------------
repo_owner = "sreesh49"
repo_name = "YT-Capstone-Project"

mlflow_tracking_uri = f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
mlflow.set_tracking_uri(mlflow_tracking_uri)

dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)

# -----------------------------
# Load model info
# -----------------------------
def load_model_info(file_path: str) -> dict:
    with open(file_path, "r") as f:
        return json.load(f)

# -----------------------------
# Register model
# -----------------------------
def register_model(model_name: str, model_info: dict):

    model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"

    # Register model
    result = mlflow.register_model(model_uri=model_uri, name=model_name)

    client = MlflowClient()

    # Wait until model version is ready (important for DagsHub)
    client.transition_model_version_stage(
        name=model_name,
        version=result.version,
        stage="Staging"
    )

    logging.info(f"Model {model_name} v{result.version} registered → Staging")

# -----------------------------
# Main
# -----------------------------
def main():
    model_info_path = "reports/experiment_info.json"
    model_info = load_model_info(model_info_path)

    register_model(
        model_name="my_model",
        model_info=model_info
    )

if __name__ == "__main__":
    main()