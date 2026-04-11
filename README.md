# Mini AI Analyst (AaaS)

> **Mini AI Analyst as a Service** - Upload a CSV, profile the dataset, train an ML model, and predict. All from your browser. 🧠

This is a lightweight application designed to auto-profile structured tabular payloads and perform automated model training (Classification/Regression) using Scikit-Learn and FastAPI.

## ✨ Features
1. **Upload**: Drag & drop or browse to upload a `.csv` dataset (max 50 MB by default).
2. **Profile**: Analyze column dependencies, auto-detect data types, pinpoint percentiles of missing/null items, and compute the presence of outliers.
3. **Train Model**: Choose a target column, let our analyzer detect if it's a classification or regression problem, and train a model out of the box. Small datasets trigger simpler models (`LogisticRegression`, `LinearRegression`); larger ones utilize ensemble models (`RandomForest`).
4. **Predict**: Provide a JSON payload array, run inferences against the joblib-persisted model UUID, and fetch outcome probabilities.
5. **Summary**: A natural-language-like summary that integrates dataset statistics with core model metrics like RMSE, R², Accuracy, etc.

## 🏗️ Architecture

- **Backend**: FastAPI (Python 3.9+)
- **ML Services**: Pandas, NumPy, Scikit-learn
- **Frontend**: Plain HTML, CSS, Vanilla JS
- **Persistence**: Local Filesystem (PostgreSQL placeholder included for later V2 integration)

## 🚀 QuickStart

### Prerequisites
Make sure you have Python 3.9+ installed and a virtual environment active.

### 1. Installation

Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Running the Server

Start the local server using Uvicorn:
```bash
uvicorn app.main:app --reload --port 8000
```
Then, point your web browser to [http://localhost:8000](http://localhost:8000).

### 3. Running Tests

We've bundled `pytest` suites hitting every route handler.
```bash
pytest tests/ -v
```

## 📁 Repository Structure

```
├── app/
│   ├── api/             # API Router definitions
│   ├── core/            # Config, Utilities, Custom Logger
│   ├── models/          # ML preprocessing, training, and evaluation logic
│   ├── schemas/         # Pydantic schema validation interfaces
│   ├── services/        # Central business logic handlers
│   ├── storage/         # Persisted `.csv` and `.pkl` joblib instances
│   └── main.py          # Application Factory Setup
├── frontend/
│   └── [HTML, CSS, JS]
├── tests/               # Pytest suite
├── .gitignore           
├── .env.example         # Example Environment mapping
└── requirements.txt     
```

## 🔐 Security Considerations
Please note `app/storage/models` and `app/storage/uploads` are marked in `.gitignore` to prevent you from mistakenly pushing `.csv` datasets and generated `.pkl` models to version control. Let's keep data secure!
