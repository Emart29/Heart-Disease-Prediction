# ❤️ Heart Disease Prediction

[![CI/CD Pipeline](https://github.com/Emart29/heart-disease-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/Emart29/heart-disease-prediction/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **production-ready** machine learning system for heart disease risk assessment, featuring a FastAPI REST API, Streamlit dashboard, SHAP explainability, MLflow experiment tracking, and Dockerized deployment.

**From notebook → production-grade application with full CI/CD.**

---

## 🎯 Project Overview

This project predicts the likelihood of heart disease based on 13 clinical features and provides **explainable predictions** with SHAP values.

### Key Features

| Feature | Description |
|---------|-------------|
| 🚀 **REST API** | FastAPI service with auto-generated docs |
| 🖥️ **Web Interface** | Interactive Streamlit dashboard |
| 🔍 **Explainable AI** | SHAP-based feature importance |
| 📊 **MLOps Ready** | MLflow tracking, versioned models |
| 🐳 **Containerized** | Docker support for consistent deployment |
| ✅ **Well-Tested** | pytest + property-based tests |
| 🔄 **CI/CD** | GitHub Actions for linting, tests, and Docker build |

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 88.5% |
| **ROC-AUC** | 0.954 |
| **Precision** | 86.2% |
| **Recall** | 89.3% |
| **F1 Score** | 87.7% |

- **Model:** Random Forest Classifier
- **Features:** 13 clinical attributes + 22 engineered features
- **Dataset:** [UCI Heart Disease](https://archive.ics.uci.edu/ml/datasets/heart+disease)

---

## 🏗️ Architecture

```
User Interface
 ┌───────────────┐       ┌───────────────┐
 │ Streamlit App │       │ Swagger UI    │
 │ (Port 8501)   │       │ (Port 8000)   │
 └─────┬─────────┘       └─────┬─────────┘
       │                       │
       ▼                       ▼
 API Layer
 ┌─────────────────────────────────────────┐
 │ FastAPI Server                          │
 │ • /health    • /model-info   • /predict │
 └─────┬───────────────────────────────────┘
       │
 ML Pipeline
 ┌──────────────┐  ┌─────────────┐  ┌─────────────────┐
 │ Feature Eng  │→ │ Model       │→ │ SHAP Explainer  │
 └──────────────┘  └─────────────┘  └─────────────────┘
       │
 MLOps Layer
 ┌─────────────────────────────────────────┐
 │ MLflow Registry                         │
 │ • Experiment tracking                   │
 │ • Model versioning                      │
 │ • Artifacts storage                     │
 └─────────────────────────────────────────┘

 CI/CD Pipeline (GitHub Actions)
 ┌─────────────────────────────────────────┐
 │ On Push/PR:                             │
 │ • Lint (black, flake8)                  │
 │ • Test (pytest)                         │
 │ • Build (Docker)                        │
 └─────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- pip or conda
- Docker (optional)

### Local Installation

```bash
# Clone repo
git clone https://github.com/Emart29/heart-disease-prediction.git
cd heart-disease-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
📍 API Docs: http://localhost:8000/docs

### Run Streamlit App

```bash
streamlit run app/streamlit_app.py
```
📍 Dashboard: http://localhost:8501

---

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

```bash
# Build and start all services
docker-compose up --build

# Run in detached mode
docker-compose up -d --build

# Stop services
docker-compose down
```

### Using Docker Directly

```bash
# Build image
docker build -t heart-disease-prediction .

# Run API
docker run -p 8000:8000 heart-disease-prediction

# Run Streamlit
docker run -p 8501:8501 heart-disease-prediction \
  streamlit run app/streamlit_app.py --server.port=8501 --server.address=0.0.0.0
```

---

## 🔄 CI/CD Pipeline

This project includes a **fully automated CI/CD pipeline** using GitHub Actions:

| Step | Tool | Purpose |
|------|------|---------|
| **Linting** | black, flake8 | Code formatting & style |
| **Testing** | pytest | Unit & property-based tests |
| **Coverage** | pytest-cov | Code coverage reporting |
| **Build** | Docker | Container build verification |

### Pipeline Triggers
- ✅ On push to `main`
- ✅ On pull requests
- ✅ Manual dispatch

### Running CI Locally

```bash
# Format code
black src/ api/ tests/

# Lint
flake8 src/ api/ tests/

# Run tests
pytest --cov=src --cov=api --cov-report=html
```

---

## 🧪 Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=src --cov=api --cov-report=html

# Run specific test file
pytest tests/test_api.py -v
```

**Test Coverage:**
- ✅ Model loading & prediction
- ✅ Feature engineering
- ✅ API endpoints
- ✅ Data validation
- ✅ Property-based tests (Hypothesis)

---

## 📡 API Endpoints

### Health Check
```http
GET /health
```
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Model Info
```http
GET /model-info
```
```json
{
  "version": "1.0.0",
  "model_type": "RandomForestClassifier",
  "features": ["age", "sex", "cp", ...],
  "training_date": "2025-12-29",
  "metrics": {
    "accuracy": 0.885,
    "roc_auc": 0.954
  }
}
```

### Predict
```http
POST /predict
```
```json
{
  "age": 55,
  "sex": 1,
  "cp": 3,
  "trestbps": 140,
  "chol": 250,
  "fbs": 0,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 1.5,
  "slope": 2,
  "ca": 0,
  "thal": 3
}
```
**Response:**
```json
{
  "prediction": 1,
  "probability": 0.6523,
  "risk_level": "Medium",
  "feature_importance": {
    "age": 0.0234,
    "thalach": -0.0456,
    "ca": 0.0891
  }
}
```

---

## 📁 Project Structure

```
heart_disease_prediction/
├── .github/
│   └── workflows/
│       └── ci.yml          # CI/CD pipeline
├── src/                    # Source code
│   ├── config.py
│   ├── data/
│   ├── features/
│   │   └── engineering.py
│   ├── models/
│   │   ├── train.py
│   │   └── predict.py
│   └── validation/
│       └── schemas.py
├── api/                    # FastAPI
│   └── main.py
├── app/                    # Streamlit
│   └── streamlit_app.py
├── tests/                  # Unit & property-based tests
├── models/                 # Saved model artifacts
├── data/                   # Datasets
├── mlruns/                 # MLflow artifacts
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 🔬 MLflow Tracking

```bash
# Train a new model with MLflow
python -m src.models.train --experiment-name "heart_disease_v2" --n-estimators 200

# Launch MLflow UI
mlflow ui --port 5000
```

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| **ML/Data** | scikit-learn, pandas, numpy, SHAP |
| **API** | FastAPI, Pydantic, uvicorn |
| **Web** | Streamlit, Plotly |
| **MLOps** | MLflow |
| **Testing** | pytest, hypothesis |
| **CI/CD** | GitHub Actions |
| **Container** | Docker, Docker Compose |

---

## 📄 License

MIT License – see [LICENSE](LICENSE) file.

---

## 📬 Contact

- **LinkedIn:** [linkedin.com/in/nwangumaemmanuel](https://linkedin.com/in/nwangumaemmanuel)
- **Email:** nwangumaemmanuel29@gmail.com
- **GitHub Issues:** [Open an issue](https://github.com/Emart29/heart-disease-prediction/issues)

---

## ⭐ Star This Repo

If you found this project helpful, please give it a star! It helps others discover the project.

[![Star History](https://img.shields.io/github/stars/Emart29/heart-disease-prediction?style=social)](https://github.com/Emart29/heart-disease-prediction)
