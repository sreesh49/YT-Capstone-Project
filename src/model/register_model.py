import os
import json
import mlflow
from mlflow.tracking import MlflowClient
import dagshub
from src.logger import logging


# =========================
# MLflow Setup
# =========================
def setup_mlflow():

    token = os.getenv("DAGSHUB_TOKEN")
    if not token:
        raise EnvironmentError("DAGSHUB_TOKEN not set")

    # Required for DagsHub MLflow auth
    os.environ["MLFLOW_TRACKING_USERNAME"] = "token"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    tracking_uri = "https://dagshub.com/sreesh49/YT-Capstone-Project.mlflow"

    # IMPORTANT: set both env + explicit call
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    mlflow.set_tracking_uri(tracking_uri)

    dagshub.init(
        repo_owner="sreesh49",
        repo_name="YT-Capstone-Project",
        mlflow=True
    )

    logging.info("MLflow setup completed")


# =========================
# Load model info
# =========================
def load_model_info(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")

    with open(path, "r") as f:
        return json.load(f)


# =========================
# Register model (SAFE)
# =========================
def register_model(model_name, run_id, model_path):

    client = MlflowClient()

    model_uri = f"runs:/{run_id}/{model_path}"

    logging.info(f"Registering model: {model_uri}")

    # Register model
    result = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )

    version = result.version

    logging.info(f"Model registered: {model_name} v{version}")

    # SAFE staging transition (CI-friendly fallback)
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Staging"
        )
        logging.info("Model moved to Staging")
    except Exception as e:
        logging.warning(f"Stage transition skipped: {e}")


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