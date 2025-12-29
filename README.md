# ❤️ Heart Disease Prediction

A production-ready machine learning application for heart disease risk assessment, featuring a REST API, interactive web interface, and MLOps best practices.

## 🎯 Project Overview

This project demonstrates end-to-end ML engineering skills by transforming a Jupyter notebook analysis into a deployable, production-grade application. It predicts the likelihood of heart disease based on 13 clinical features and provides explainable predictions using SHAP values.

### Key Features

- **REST API**: FastAPI-powered prediction service with automatic documentation
- **Web Interface**: Interactive Streamlit dashboard for easy predictions
- **Explainable AI**: SHAP-based feature importance for every prediction
- **MLOps Ready**: MLflow experiment tracking and model versioning
- **Containerized**: Docker support for consistent deployment
- **Well-Tested**: Property-based and unit tests with pytest

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                           │
│  ┌─────────────────────┐       ┌─────────────────────────────┐ │
│  │   Streamlit App     │       │      Swagger UI             │ │
│  │   (Port 8501)       │       │      (Port 8000/docs)       │ │
│  └──────────┬──────────┘       └──────────────┬──────────────┘ │
└─────────────┼──────────────────────────────────┼───────────────┘
              │                                  │
              ▼                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                         API Layer                               │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    FastAPI Server                           ││
│  │  • /health - Health check                                   ││
│  │  • /model-info - Model metadata                             ││
│  │  • /predict - Heart disease prediction                      ││
│  └──────────────────────────┬──────────────────────────────────┘│
│                             │                                   │
│  ┌──────────────────────────▼──────────────────────────────────┐│
│  │              Pydantic Data Validation                       ││
│  └──────────────────────────┬──────────────────────────────────┘│
└─────────────────────────────┼───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       ML Pipeline                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │    Feature      │  │     Model       │  │      SHAP       │ │
│  │   Engineering   │──│   Predictor     │──│   Explainer     │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        MLOps Layer                              │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                   MLflow Registry                           ││
│  │  • Experiment tracking  • Model versioning  • Artifacts     ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 88.5% |
| ROC-AUC | 0.954 |
| Precision | 86.2% |
| Recall | 89.3% |
| F1 Score | 87.7% |

The model uses a Random Forest Classifier trained on the UCI Heart Disease dataset with 22 engineered features derived from 13 clinical attributes.

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- pip or conda

### Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Emart29/heart-disease-prediction.git
   cd heart-disease-prediction
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the API server**
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```

5. **Start the Streamlit app** (in a new terminal)
   ```bash
   streamlit run app/streamlit_app.py
   ```

6. **Access the applications**
   - API Documentation: http://localhost:8000/docs
   - Web Interface: http://localhost:8501

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
# Build the image
docker build -t heart-disease-prediction .

# Run API server
docker run -p 8000:8000 heart-disease-prediction

# Run Streamlit app
docker run -p 8501:8501 heart-disease-prediction \
  streamlit run app/streamlit_app.py --server.port=8501 --server.address=0.0.0.0
```

### Service Ports

| Service | Port | URL |
|---------|------|-----|
| FastAPI | 8000 | http://localhost:8000 |
| Swagger UI | 8000 | http://localhost:8000/docs |
| Streamlit | 8501 | http://localhost:8501 |

## 📡 API Documentation

### Endpoints

#### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

#### Model Information
```http
GET /model-info
```

**Response:**
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

#### Predict Heart Disease Risk
```http
POST /predict
Content-Type: application/json
```

**Request Body:**
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
    "ca": 0.0891,
    ...
  }
}
```

### Input Features

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| age | int | 20-100 | Patient age in years |
| sex | int | 0, 1 | 0=Female, 1=Male |
| cp | int | 1-4 | Chest pain type |
| trestbps | int | 80-200 | Resting blood pressure (mm Hg) |
| chol | int | 100-600 | Serum cholesterol (mg/dl) |
| fbs | int | 0, 1 | Fasting blood sugar > 120 mg/dl |
| restecg | int | 0-2 | Resting ECG results |
| thalach | int | 60-220 | Maximum heart rate achieved |
| exang | int | 0, 1 | Exercise induced angina |
| oldpeak | float | 0.0-7.0 | ST depression |
| slope | int | 1-3 | Slope of peak exercise ST segment |
| ca | int | 0-3 | Number of major vessels |
| thal | int | 3, 6, 7 | Thalassemia type |

### Example: cURL Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### Example: Python Request

```python
import requests

patient_data = {
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

response = requests.post(
    "http://localhost:8000/predict",
    json=patient_data
)

result = response.json()
print(f"Risk Level: {result['risk_level']}")
print(f"Probability: {result['probability']:.1%}")
```

## 📁 Project Structure

```
heart_disease_prediction/
├── src/                          # Source code
│   ├── config.py                 # Configuration management
│   ├── data/                     # Data loading utilities
│   ├── features/                 # Feature engineering
│   │   └── engineering.py
│   ├── models/                   # Model training and prediction
│   │   ├── train.py              # MLflow training script
│   │   └── predict.py            # Prediction with SHAP
│   └── validation/               # Input validation
│       └── schemas.py            # Pydantic schemas
├── api/                          # FastAPI application
│   └── main.py
├── app/                          # Streamlit application
│   └── streamlit_app.py
├── tests/                        # Test suite
│   ├── test_api.py
│   ├── test_features.py
│   ├── test_predictor.py
│   └── test_validation.py
├── data/                         # Dataset files
├── models/                       # Trained model artifacts
├── mlruns/                       # MLflow experiment tracking
├── docs/                         # Documentation
│   └── MODEL_CARD.md
├── reports/                      # Generated figures (EDA, SHAP plots)
├── data_collection.ipynb         # Data collection notebook
├── data_preprocessing.ipynb      # Data preprocessing notebook
├── exploratory_data_analytics.ipynb  # EDA notebook
├── model_developmernt.ipynb      # Model development notebook
├── Dockerfile                    # Container definition
├── docker-compose.yml            # Multi-service orchestration
├── requirements.txt              # Python dependencies
└── README.md
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov=api --cov-report=html

# Run specific test file
pytest tests/test_api.py -v

# Run property-based tests
pytest tests/test_validation.py -v
```

## 🔬 MLflow Experiment Tracking

Train a new model with MLflow tracking:

```bash
python -m src.models.train --experiment-name "heart_disease_v2" --n-estimators 200
```

View experiments:

```bash
mlflow ui --port 5000
```

Then open http://localhost:5000 to explore experiment runs, compare metrics, and manage model versions.

## 📖 Documentation

- [Model Card](docs/MODEL_CARD.md) - Detailed model documentation including performance, limitations, and ethical considerations

## 🛠️ Technologies Used

- **ML/Data**: scikit-learn, pandas, numpy, SHAP
- **API**: FastAPI, Pydantic, uvicorn
- **Web**: Streamlit, Plotly
- **MLOps**: MLflow
- **Testing**: pytest, hypothesis
- **Containerization**: Docker, Docker Compose

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the Heart Disease dataset
- Cleveland Clinic Foundation for the original data collection
- The scikit-learn, FastAPI, and Streamlit communities

## 📬 Contact

For questions or feedback, please open an issue in this repository.

---

*Built as a portfolio project demonstrating production ML engineering practices.*
#   H e a r t - D i s e a s e - P r e d i c t i o n  
 