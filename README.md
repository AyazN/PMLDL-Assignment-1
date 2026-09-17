# PMLDL Assignment 1 — MLOps Pipeline

## Overview

This project implements an end-to-end MLOps pipeline for **red wine quality classification**.

The pipeline consists of three stages:

1. **Data Engineering** — cleaning, outlier removal, target transformation, and train/test splitting using DVC.
2. **Model Engineering** — feature scaling, Random Forest training, evaluation, and MLflow experiment tracking.
3. **Deployment** — FastAPI model API and Streamlit web application running in separate Docker containers.

The complete pipeline is automated to run every **5 minutes**.

---

## Pipeline

```text
Wine Quality Dataset
        │
        ▼
   Data Cleaning
        │
        ▼
 Train / Test Split
        │
        ▼
 Feature Scaling
        │
        ▼
 Random Forest Model
        │
        ├──► Accuracy / F1
        │
        └──► MLflow
        │
        ▼
   model.joblib
        │
        ▼
   FastAPI API
        │
        ▼
 Streamlit Web App
```

---

## Project Structure

```text
PMLDL-Assignment-1/
├── code/
│   ├── datasets/
│   │   └── prepare_data.py
│   ├── models/
│   │   └── train.py
│   └── deployment/
│       ├── api/
│       │   ├── main.py
│       │   ├── Dockerfile
│       │   └── requirements.txt
│       ├── app/
│       │   ├── app.py
│       │   ├── Dockerfile
│       │   └── requirements.txt
│       └── docker-compose.yml
├── data/
│   ├── raw/
│   │   └── winequality-red.csv
│   └── processed/
│       ├── train.csv
│       └── test.csv
├── models/
│   └── model.joblib
├── metrics/
│   └── metrics.json
├── dvc.yaml
├── run_pipeline.py
├── requirements.txt
└── README.md
```

---

## Dataset

The project uses the **UCI Wine Quality — Red Wine** dataset.

The dataset is stored at:

```text
data/raw/winequality-red.csv
```

Wine quality is converted into a binary classification target:

```text
quality >= 6  →  Good wine (1)
quality < 6   →  Not good wine (0)
```

---

## Stage 1 — Data Engineering

`code/datasets/prepare_data.py` performs:

* Loading the raw dataset
* Duplicate removal
* Missing-value removal
* IQR-based outlier removal
* Binary target creation
* Stratified train/test split

The processed datasets are saved as:

```text
data/processed/train.csv
data/processed/test.csv
```

DVC manages the data-processing pipeline through `dvc.yaml`.

---

## Stage 2 — Model Engineering

`code/models/train.py` performs:

* Feature scaling using `StandardScaler`
* Random Forest classification
* Model evaluation using:

  * Accuracy
  * F1 Score
* MLflow experiment tracking
* Model packaging with Joblib

The trained model is saved to:

```text
models/model.joblib
```

Metrics are saved to:

```text
metrics/metrics.json
```

MLflow tracks the experiment under:

```text
wine-quality
```

using a local SQLite database.

---

## Stage 3 — Deployment

The trained model is deployed using two separate Docker containers.

### FastAPI

The FastAPI service provides the prediction endpoint:

```text
POST /predict
```

API:

```text
http://localhost:8000
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

### Streamlit

The Streamlit application provides input fields for the wine characteristics and displays the prediction returned by the API.

Web application:

```text
http://localhost:8501
```

Docker Compose is used to run the API and web application as separate services.

---

## Automation

`run_pipeline.py` automatically executes the pipeline every 5 minutes.

The pipeline runs:

```text
DVC pipeline
    ↓
Model training
    ↓
Docker build/deployment
```

The interval is configured as:

```python
INTERVAL = 300
```

---

## Installation and Usage

### 1. Install dependencies

This project uses global Python 3:

```bash
python3 -m pip install -r requirements.txt
```

### 2. Initialize DVC

If DVC has not already been initialized:

```bash
git init
dvc init
```

### 3. Run the complete pipeline

```bash
python3 run_pipeline.py
```

Alternatively, Docker can be started directly with:

```bash
docker compose -f code/deployment/docker-compose.yml up --build
```

### 4. Open the application

After the containers start:

```text
Streamlit:  http://localhost:8501
FastAPI:    http://localhost:8000
API Docs:   http://localhost:8000/docs
```

---

## Technologies

* **Python 3**
* **Pandas**
* **Scikit-learn**
* **DVC**
* **MLflow**
* **Joblib**
* **FastAPI**
* **Streamlit**
* **Docker / Docker Compose**
* **Git / GitHub**

---

## Assignment Requirements

| Requirement         | Implementation                              |
| ------------------- | ------------------------------------------- |
| Data engineering    | Cleaning, outlier removal, train/test split |
| Pipeline management | DVC                                         |
| Feature engineering | StandardScaler                              |
| Model training      | Random Forest                               |
| Model evaluation    | Accuracy and F1 Score                       |
| Experiment tracking | MLflow                                      |
| Model packaging     | Joblib                                      |
| API deployment      | FastAPI                                     |
| Web application     | Streamlit                                   |
| Separate containers | Docker Compose                              |
| Automation          | `run_pipeline.py`, every 5 minutes          |
| Version control     | Git/GitHub                                  |
