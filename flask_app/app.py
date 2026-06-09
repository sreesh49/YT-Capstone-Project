
from flask import Flask, render_template, request
import mlflow
import pickle
import os
import pandas as pd
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CollectorRegistry,
    CONTENT_TYPE_LATEST
)
import time
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import re
import dagshub
import warnings

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# =========================================================
# TEXT PREPROCESSING
# =========================================================

def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)


def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)


def removing_numbers(text):
    return ''.join([char for char in text if not char.isdigit()])


def lower_case(text):
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)


def removing_punctuations(text):
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub('\s+', ' ', text).strip()
    return text


def removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)


def normalize_text(text):
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)

    return text

# =========================================================
# DAGSHUB + MLFLOW SETUP
# =========================================================

repo_owner = "sreesh49"
repo_name = "YT-Capstone-Project"

dagshub_token = os.getenv("DAGSHUB_TOKEN")

if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN is not set")

    os.environ["DAGSHUB_USER_TOKEN"] = dagshub_token

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    mlflow.set_tracking_uri(
        f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
    )

    dagshub.init(
        repo_owner=repo_owner,
        repo_name=repo_name,
        mlflow=True,
        auth_token=dagshub_token
    )

    print("DAGsHub connected successfully")

else:
    print("No DAGSHUB_TOKEN found → running in local mode")

# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)

# =========================================================
# PROMETHEUS METRICS
# =========================================================

registry = CollectorRegistry()

REQUEST_COUNT = Counter(
    "app_request_count",
    "Total number of requests to the app",
    ["method", "endpoint"],
    registry=registry
)

REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds",
    "Latency of requests in seconds",
    ["endpoint"],
    registry=registry
)

PREDICTION_COUNT = Counter(
    "model_prediction_count",
    "Count of predictions for each class",
    ["prediction"],
    registry=registry
)

# =========================================================
# MODEL LOADING
# =========================================================

model = None

try:

    print("Trying to load model from MLflow Registry...")

    model_name = "my_model"

    client = mlflow.MlflowClient()

    latest_version = client.get_latest_versions(
        model_name,
        stages=["Production"]
    )

    if not latest_version:
        latest_version = client.get_latest_versions(
            model_name,
            stages=["None"]
        )

    model_version = latest_version[0].version

    model_uri = f"models:/{model_name}/{model_version}"

    print(f"Fetching model from: {model_uri}")

    model = mlflow.pyfunc.load_model(model_uri)

    print("MLflow model loaded successfully")

except Exception as e:

    print(f"MLflow model loading failed: {e}")

    print("Loading local model.pkl instead...")

    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)

    print("Local model loaded successfully")

# =========================================================
# LOAD VECTORIZER
# =========================================================

with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

print("Vectorizer loaded successfully")

# =========================================================
# ROUTES
# =========================================================

@app.route("/")
def home():

    REQUEST_COUNT.labels(
        method="GET",
        endpoint="/"
    ).inc()

    start_time = time.time()

    response = render_template(
        "index.html",
        result=None
    )

    REQUEST_LATENCY.labels(
        endpoint="/"
    ).observe(time.time() - start_time)

    return response


@app.route("/predict", methods=["POST"])
def predict():

    REQUEST_COUNT.labels(
        method="POST",
        endpoint="/predict"
    ).inc()

    start_time = time.time()

    text = request.form["text"]

    # preprocess text
    text = normalize_text(text)

    # vectorize
    features = vectorizer.transform([text])

    features_df = pd.DataFrame(
        features.toarray(),
        columns=[str(i) for i in range(features.shape[1])]
    )

    # prediction
    result = model.predict(features_df)

    prediction = result[0]

    PREDICTION_COUNT.labels(
        prediction=str(prediction)
    ).inc()

    REQUEST_LATENCY.labels(
        endpoint="/predict"
    ).observe(time.time() - start_time)

    return render_template(
        "index.html",
        result=prediction
    )


@app.route("/metrics", methods=["GET"])
def metrics():

    return (
        generate_latest(registry),
        200,
        {"Content-Type": CONTENT_TYPE_LATEST}
    )

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )
```
