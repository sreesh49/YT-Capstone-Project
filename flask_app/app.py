from flask import Flask, render_template, request
import mlflow
import pickle
import os
import pandas as pd
import numpy as np
import time
import re
import string

from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from mlflow.tracking import MlflowClient

import warnings
warnings.filterwarnings("ignore")

# =========================
# TEXT PREPROCESSING
# =========================
def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(w) for w in text.split()])


def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    return " ".join([w for w in str(text).split() if w not in stop_words])


def removing_numbers(text):
    return ''.join([c for c in text if not c.isdigit()])


def lower_case(text):
    return " ".join([w.lower() for w in text.split()])


def removing_punctuations(text):
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    return re.sub('\s+', ' ', text).strip()


def removing_urls(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)


def normalize_text(text):
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)
    return text


# =========================
# FLASK APP
# =========================
app = Flask(__name__)

registry = CollectorRegistry()

REQUEST_COUNT = Counter(
    "app_request_count",
    "Total requests",
    ["method", "endpoint"],
    registry=registry
)

REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds",
    "Latency",
    ["endpoint"],
    registry=registry
)

PREDICTION_COUNT = Counter(
    "model_prediction_count",
    "Prediction count",
    ["prediction"],
    registry=registry
)


# =========================
# MLflow CONFIG (CLEAN)
# =========================
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "https://dagshub.com/sreesh49/YT-Capstone-Project.mlflow"
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# =========================
# LOAD MODEL FROM REGISTRY
# =========================
model_name = "my_model"

try:
    client = MlflowClient()

    versions = client.search_model_versions(f"name='{model_name}'")

    if not versions:
        raise Exception("No model found in registry")

    latest_version = max(int(v.version) for v in versions)

    model_uri = f"models:/{model_name}/{latest_version}"

    print(f"Loading model version: {latest_version}")

    model = mlflow.pyfunc.load_model(model_uri)

    print("Model loaded successfully!")

except Exception as e:
    print(f"Model loading failed: {e}")
    model = None


# =========================
# LOAD VECTORIZER
# =========================
vectorizer_path = "models/vectorizer.pkl"

if not os.path.exists(vectorizer_path):
    raise FileNotFoundError("Vectorizer missing")

with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)


# =========================
# ROUTES
# =========================
@app.route("/")
def home():
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start = time.time()

    response = render_template("index.html", result=None)

    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start)
    return response


@app.route("/predict", methods=["POST"])
def predict():
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start = time.time()

    text = request.form["text"]
    text = normalize_text(text)

    features = vectorizer.transform([text])
    features_df = pd.DataFrame(features.toarray())

    prediction = model.predict(features_df)[0]

    PREDICTION_COUNT.labels(prediction=str(prediction)).inc()

    REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start)

    return render_template("index.html", result=prediction)


@app.route("/metrics")
def metrics():
    return generate_latest(registry), 200, {
        "Content-Type": CONTENT_TYPE_LATEST
    }


# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)