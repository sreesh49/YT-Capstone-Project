import os
import json
import mlflow
from mlflow.tracking import MlflowClient
import dagshub
from src.logger import logging


# =========================
# MLflow Setup (CI + DVC SAFE)
# =========================
def setup_mlflow():

    token = os.getenv("DAGSHUB_TOKEN")

    if not token:
        raise EnvironmentError("DAGSHUB_TOKEN not set")

    # ✅ REQUIRED for DagsHub MLflow auth
    os.environ["MLFLOW_TRACKING_USERNAME"] = "token"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    # ✅ IMPORTANT: always set explicitly
    mlflow.set_tracking_uri(
        "https://dagshub.com/sreesh49/YT-Capstone-Project.mlflow"
    )

    dagshub.init(
        repo_owner="sreesh49",
        repo_name="YT-Capstone-Project",
        mlflow=True
    )


# =========================
# Load experiment info
# =========================
def load_model_info(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")

    with open(path, "r") as f:
        return json.load(f)


# =========================
# Register model safely
# =========================
def register_model(model_name, run_id, model_path):

    client = MlflowClient()

    model_uri = f"runs:/{run_id}/{model_path}"

    logging.info(f"Registering model from: {model_uri}")

    # Register model
    result = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )

    # ✅ SAFE: alias only if registry supports it
    try:
        client.set_registered_model_alias(
            name=model_name,
            alias="staging",
            version=result.version
        )
        logging.info("Alias 'staging' set successfully")
    except Exception as e:
        logging.warning(f"Alias setup skipped: {e}")

    logging.info(
        f"Model {model_name} version {result.version} registered successfully"
    )


# =========================
# Main
# =========================
def main():

    setup_mlflow()

    info = load_model_info("reports/experiment_info.json")

    register_model(
        model_name="my_model",
        run_id=info["run_id"],
        model_path=info["model_path"]
    )


if __name__ == "__main__":
    main()