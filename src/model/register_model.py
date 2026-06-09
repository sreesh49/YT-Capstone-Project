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

import sys
import os
import json
import mlflow
import dagshub
from mlflow.tracking import MlflowClient
from src.logger import logging
import warnings

warnings.filterwarnings("ignore")

# -----------------------------
# CI guard (CRITICAL)
# -----------------------------
if os.getenv("CI") == "true":
    print("CI detected → skipping model registration")
    exit(0)

# -----------------------------
# Auth
# -----------------------------
dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN is not set")
os.environ["DAGSHUB_USER_TOKEN"] = dagshub_token
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# -----------------------------
# Setup
# -----------------------------
repo_owner = "sreesh49"
repo_name = "YT-Capstone-Project"

mlflow_tracking_uri = f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
mlflow.set_tracking_uri(mlflow_tracking_uri)

dagshub.init(
    repo_owner=repo_owner,
    repo_name=repo_name,
    mlflow=True,
    
)

# -----------------------------
def load_model_info(file_path: str):
    with open(file_path, "r") as f:
        return json.load(f)

# -----------------------------
def register_model(model_name: str, model_info: dict):

    model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"

    result = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )

    client = MlflowClient()

    client.transition_model_version_stage(
        name=model_name,
        version=result.version,
        stage="Staging"
    )

    logging.info(f"Model registered → {model_name} v{result.version}")

# -----------------------------
def main():
    model_info = load_model_info("reports/experiment_info.json")

    register_model(
        model_name="my_model",
        model_info=model_info
    )

if __name__ == "__main__":
    main()