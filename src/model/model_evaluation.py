import sys
import os

# Add project root to Python path - MUST BE BEFORE src imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import mlflow
import mlflow.sklearn
import dagshub
from src.logger import logging


dagshub_token = os.getenv("DAGSHUB_TOKEN")

if os.getenv("CI") == "true":
    print("CI detected → skipping DAGsHub tracking completely")

elif dagshub_token:
    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    dagshub.init(
        repo_owner='sreesh49',
        repo_name='YT-Capstone-Project',
        mlflow=True,
        
    )
else:
    print("No DAGSHUB_TOKEN → skipping tracking")


def load_model(file_path: str):
    """Load the trained model from a file."""
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logging.info('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model: %s', e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise

def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate the model and return the evaluation metrics."""
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'auc': float(auc)
        }
        logging.info(f'Model evaluation metrics calculated: {metrics_dict}')
        return metrics_dict
    except Exception as e:
        logging.error('Error during model evaluation: %s', e)
        raise

def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logging.info('Metrics saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the metrics: %s', e)
        raise

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logging.info('Model info saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model info: %s', e)
        raise

def main():
    try:
        # Create reports directory if it doesn't exist
        os.makedirs('reports', exist_ok=True)
        
        logging.info("Starting model evaluation with MLflow tracking...")
        logging.info(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
        
        # Set experiment
        mlflow.set_experiment("sentiment-analysis")
        logging.info("MLflow experiment set to 'sentiment-analysis'")
        
        with mlflow.start_run() as run:
            logging.info(f"Started MLflow run with ID: {run.info.run_id}")
            
            # Log run metadata
            mlflow.log_param("model_type", "LogisticRegression")
            mlflow.log_param("data_source", "S3 Bucket")
            
            # Load model and data
            clf = load_model('./models/model.pkl')
            test_data = load_data('./data/processed/test_bow.csv')
            
            X_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values
            
            logging.info(f"Test data shape: {X_test.shape}")
            mlflow.log_param("test_samples", X_test.shape[0])
            mlflow.log_param("features", X_test.shape[1])
            
            # Evaluate model
            metrics = evaluate_model(clf, X_test, y_test)
            
            # Save metrics locally
            save_metrics(metrics, 'reports/metrics.json')
            
            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
                logging.info(f"Logged metric: {metric_name} = {metric_value:.4f}")
            
            # Log model parameters to MLflow
            if hasattr(clf, 'get_params'):
                params = clf.get_params()
                for param_name, param_value in params.items():
                    # Only log primitive parameters
                    if not param_name.startswith('_') and isinstance(param_value, (str, int, float, bool)):
                        mlflow.log_param(param_name, param_value)
            
            # Log model to MLflow
            mlflow.sklearn.log_model(clf, "model")
            logging.info("Model logged to MLflow")
            
            # Save model info
            save_model_info(run.info.run_id, "model", 'reports/experiment_info.json')
            
            # Log the metrics file as artifact
            mlflow.log_artifact('reports/metrics.json')
            
            # Print results
            print("\n" + "="*50)
            print("MODEL EVALUATION RESULTS")
            print("="*50)
            for metric_name, metric_value in metrics.items():
                print(f"{metric_name.upper()}: {metric_value:.4f}")
            print("="*50)
            print(f"\n✅ MLflow Run ID: {run.info.run_id}")
            print(f"✅ Tracking URI: {mlflow.get_tracking_uri()}")
            print("✅ View results at: https://dagshub.com/sreesh49/YT-Capstone-Project")
            print("✅ Metrics saved to: reports/metrics.json")
            print("✅ Model info saved to: reports/experiment_info.json")
            
            logging.info("Model evaluation completed successfully!")
            
    except Exception as e:
        logging.error('Failed to complete the model evaluation process: %s', e, exc_info=True)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()