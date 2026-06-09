
import os
import mlflow
from mlflow.tracking import MlflowClient

# ---------------------------------------------------
# CI guard
# ---------------------------------------------------
if os.getenv("CI") == "true":
    print("Running inside GitHub Actions")

# ---------------------------------------------------
# Auth
# ---------------------------------------------------
dagshub_token = os.getenv("DAGSHUB_TOKEN")

if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
os.environ["DAGSHUB_USER_TOKEN"] = dagshub_token

# ---------------------------------------------------
# MLflow Tracking URI
# ---------------------------------------------------
repo_owner = "sreesh49"
repo_name = "YT-Capstone-Project"

tracking_uri = f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"

mlflow.set_tracking_uri(tracking_uri)

print("Tracking URI:", mlflow.get_tracking_uri())

# ---------------------------------------------------
# Promote latest model
# ---------------------------------------------------
def promote_model():

    client = MlflowClient()

    model_name = "my_model"

    # Get all versions
    versions = client.search_model_versions(
        f"name='{model_name}'"
    )

    if not versions:
        raise Exception("No model versions found")

    # Sort by version number
    latest_version = sorted(
        versions,
        key=lambda x: int(x.version),
        reverse=True
    )[0]

    version_number = latest_version.version

    print(f"Promoting model version: {version_number}")

    # Move to Production
    client.transition_model_version_stage(
        name=model_name,
        version=version_number,
        stage="Production"
    )

    print(
        f"Model {model_name} version {version_number} promoted to Production"
    )


# ---------------------------------------------------
if __name__ == "__main__":
    promote_model()
