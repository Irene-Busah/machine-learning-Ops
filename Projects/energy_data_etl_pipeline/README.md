# 🔋 Energy Price Prediction ML Pipeline

A Machine Learning project to estimate electricity prices across U.S. states using installed capability and sector data. The project includes a full ML pipeline (ETL → MLflow tracking → Streamlit dashboard) built with:

- ✅ Prefect (orchestration)
- ✅ MLflow (experiment tracking and model registry)
- ✅ Scikit-learn (model training)
- ✅ Streamlit (interactive app for predictions)

---

## 🚀 Project Overview

This project automates:
- Loading raw electricity sales & capability data
- Cleaning and transforming features
- Training and tracking regression models via MLflow
- Deploying the best model in a modern, responsive Streamlit interface

---

## 📁 Project Structure
energy-pipeline/

├── data/

│ ├── electricity_sales.csv

│ └── electricity_capability_nested.json

├── src/

│ ├── ingest_data.py

│ ├── transform_data.py 

│ ├── model.py

│ ├── orchestrate_pipeline.py

├── app/

│ └── app.py 

├── mlruns/ 

├── mlflow.db

├── prefect.yaml

├── requirements.txt

└── README.md


---

## 📊 Features Used

The model is trained on the following input features:

- `stateid`: U.S. state code (categorical)
- `sectorName`: Sector type (e.g., Residential, Commercial) (categorical)
- `capability`: Installed capability in MW (numeric)

These are vectorized using `DictVectorizer` before being passed to the regression model.

---

## 📦 Installation

1. **Clone the repo**
    ```bash
    git clone https://github.com/Irene-Busah/energy-pipeline.git
    cd energy-pipeline
    ```
2. **Create a virtual environment**
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt  
    ```
3. **Run MLflow server (in a separate terminal)**
    ```bash
    mlflow server --backend-store-uri sqlite:///mlflow.db
    ```

---
## 🧪 Running the ML Pipeline
- From the project root:
    ```bash
    python src/orchestrate_pipeline.py
    ```

This will:
1. Load and preprocess the data
2. Train a model (Linear Regression or Ridge)
3. Log metrics to MLflow
4. Optionally register the model if it's better than the previous best

---
## 🌐 Streamlit App
- Run the dashboard:
```bash
streamlit run app/app.py
```

This app:

1. Loads the model from MLflow using a run_id
2. Allows users to select state, sector, and capability
3. Predicts and displays the electricity price in a modern UI

Important: The MLflow server must be running before starting the app.

---
## 👨‍💻 Author
Irene Busah - Graduate Researcher @ CMU-Africa 

🚀 Passionate about data-driven solutions in health

