# YT-Capstone-Project
This is an end to end mlops capstone project

📌 Project Overview

This project demonstrates a production-grade MLOps pipeline covering the full lifecycle of a machine learning system — from data ingestion to model deployment on AWS EKS (Kubernetes) with CI/CD automation, experiment tracking, model versioning, and monitoring.

It showcases real-world industry-level MLOps practices used in scalable ML systems.

🧠 Key Highlights
🔁 End-to-End Automated ML Pipeline (DVC + Python)add .

📊 Experiment Tracking with MLflow (Dagshub integration)
📦 Data Versioning using DVC
🧪 Modular ML pipeline (data → features → training → evaluation)
🐳 Containerization using Docker
☁️ Cloud Deployment on AWS EKS (Kubernetes)
📦 Image storage in AWS ECR
🔄 CI/CD using GitHub Actions
📡 REST API using Flask
📈 Monitoring using Prometheus + Grafana
🔐 Secure AWS IAM + Secrets Management
🏗️ System Architecture
GitHub → CI/CD Pipeline → Docker Build → AWS ECR → Kubernetes (EKS)
                              ↓
                        ML Pipeline (DVC)
                              ↓
                  MLflow (Dagshub Tracking)
                              ↓
                 Flask API Deployment (EKS LoadBalancer)
                              ↓
            Monitoring (Prometheus + Grafana)
🧰 Tech Stack
🔹 Machine Learning
Python 3.10
Scikit-learn
Pandas / NumPy
NLTK
🔹 MLOps Tools
MLflow (Experiment Tracking + Model Registry)
DVC (Data & Pipeline Versioning)
Dagshub (Remote MLflow tracking)
Cookiecutter Data Science Template
🔹 Backend / API
Flask
Gunicorn
🔹 DevOps / Cloud
Docker
AWS ECR (Container Registry)
AWS EKS (Kubernetes Cluster)
AWS S3 (Model Storage)
AWS IAM (Security)
🔹 CI/CD
GitHub Actions
🔹 Monitoring
Prometheus
Grafana
⚙️ Project Setup
1️⃣ Environment Setup
conda create -n atlas python=3.10
conda activate atlas
pip install cookiecutter
2️⃣ Project Structure
cookiecutter -c v1 https://github.com/drivendata/cookiecutter-data-science
🔬 ML Pipeline (DVC)

Pipeline stages:

Data Ingestion
Data Preprocessing
Feature Engineering
Model Training
Model Evaluation
Model Registration

Run pipeline:

dvc repro

Check status:

dvc status
📊 Experiment Tracking (MLflow + Dagshub)
Integrated MLflow with Dagshub for remote tracking
Tracks:
Parameters
Metrics
Models
Artifacts
🐳 Dockerization

Build image:

docker build -t capstone-app:latest .

Run locally:

docker run -p 5000:5000 capstone-app:latest

Push to registry:

docker push <your-dockerhub-or-ecr>/capstone-app:latest
☁️ AWS Deployment (EKS)
Create Cluster
eksctl create cluster \
--name flask-app-cluster \
--region us-east-1 \
--nodegroup-name flask-app-nodes \
--nodes 1 \
--node-type t3.small
Deploy App
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

Get LoadBalancer URL:

kubectl get svc
🔁 CI/CD Pipeline

Automated pipeline:

Code push → GitHub Actions
Build Docker image
Push to AWS ECR
Deploy to EKS
📡 Monitoring
Prometheus Setup
Scrapes Flask API metrics via LoadBalancer endpoint
Grafana Setup
Visual dashboards for:
API latency
Request metrics
System health
🔐 Security & Secrets
AWS IAM roles for ECR & S3 access
GitHub Secrets for:
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
DAGSHUB_TOKEN
📦 Features Implemented

✔ Modular ML pipeline
✔ Data versioning (DVC)
✔ Model tracking (MLflow)
✔ Docker containerization
✔ Kubernetes deployment (EKS)
✔ CI/CD automation
✔ AWS ECR integration
✔ Real-time monitoring (Prometheus + Grafana)
✔ Scalable architecture design

📈 Future Improvements
Add Terraform for Infrastructure as Code (IaC)
Add Kubernetes HPA (Auto Scaling)
Add FastAPI instead of Flask
Add Feature Store (Feast)
Add Airflow orchestration
Add Model drift detection

👨‍💻 Author

Sreesh Sreekumar
MLOps / Data Science 

