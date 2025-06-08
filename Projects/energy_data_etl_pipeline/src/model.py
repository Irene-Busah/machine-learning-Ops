"""
model.py
--------

This script trains a regression model to predict electricity prices based on state-level
capability and sector information.

Steps:
1. Load transformed data
2. Encode categorical variables with DictVectorizer
3. Train a regression model (LinearRegression)
4. Log model, parameters, and metrics using MLflow

Outputs:
- Trained model (via MLflow)
- Run artifacts with experiment tracking

Run as a standalone script or via orchestrator.
"""

# importing necessary libraries
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
import os
import pickle

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

mlflow.set_tracking_uri("http://localhost:5000")


# loading the transformed data
def load_data(file_path: str) -> pd.DataFrame:
    """Load the transformed dataset from a CSV file."""
    return pd.read_csv(file_path)


def should_register_model(new_rmse: float, experiment_name: str = "Energy-Price-Prediction") -> bool:
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        return True  # First time logging model

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.rmse ASC"],
        max_results=1
    )
    if not runs:
        return True  # No previous runs

    best_rmse = runs[0].data.metrics["rmse"]
    return new_rmse < best_rmse


def train_model(dataframe: pd.DataFrame, model_type="LinearRegression"):
    """
    Train a regression model (Linear or Ridge) and track with MLflow.

    Args:
        dataframe (pd.DataFrame): The processed dataset
        model_type (str): "LinearRegression" or "Ridge"

    Returns:
        model, dv, rmse, model_uri (None if not registered)
    """

    # Feature engineering
    categorical = ['stateid', 'sectorName']
    numerical = ['capability']
    target = 'price'

    dataframe[categorical] = dataframe[categorical].astype(str)
    dataframe = dataframe.dropna(subset=[target])

    train_df, val_df = train_test_split(dataframe, test_size=0.2, random_state=42)

    dv = DictVectorizer()
    X_train = dv.fit_transform(train_df[categorical + numerical].to_dict(orient='records'))
    X_val = dv.transform(val_df[categorical + numerical].to_dict(orient='records'))
    y_train = train_df[target].values
    y_val = val_df[target].values

    # Select and train model
    if model_type == "LinearRegression":
        model = LinearRegression()
        model.fit(X_train, y_train)

    elif model_type == "Ridge":
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)

    else:
        raise ValueError("Unsupported model type")

    # Evaluation
    y_pred = model.predict(X_val)
    rmse = root_mean_squared_error(y_val, y_pred)

    mlflow.set_experiment('Energy-Price-Prediction')

    # MLflow logging
    with mlflow.start_run(run_name=f"{model_type}-EnergyPriceModel") as run:
        mlflow.log_param("model_type", model_type)
        mlflow.log_metric("rmse", rmse)

        # log with signature and example
        input_example = [X_val[0].toarray().tolist()[0]]
        signature = infer_signature(X_val, y_val)
        mlflow.sklearn.log_model(model, artifact_path="model", input_example=input_example, signature=signature)

        # Save DictVectorizer
        os.makedirs("data/processed", exist_ok=True)
        with open("data/processed/dv.pkl", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("data/processed/dv.pkl")

        # Conditional model registration
        if should_register_model(rmse, experiment_name="Energy-Price-Prediction"):
            model_uri = f"runs:/{run.info.run_id}/model"
            mlflow.register_model(model_uri=model_uri, name="energy_price_model")
            print("✅ New best model registered.")
        else:
            model_uri = None
            print("❌ Current model did not outperform existing best model.")

    print(f"{model_type} model trained. RMSE: {rmse:.2f}")
    return model, dv, rmse, model_uri




def get_best_model_uri(experiment_name="Energy-Price-Prediction"):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    best_run = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.rmse ASC"],
        max_results=1
    )[0]
    return f"runs:/{best_run.info.run_id}/model"


def save_best_model_locally(model_uri, output_dir="app/model"):
    model = mlflow.sklearn.load_model(model_uri)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "model.pkl"), "wb") as f_out:
        pickle.dump(model, f_out)

    # Copy DictVectorizer
    source_dv_path = "data/processed/dv.pkl"
    dest_dv_path = os.path.join(output_dir, "dv.pkl")
    if os.path.exists(source_dv_path):
        with open(source_dv_path, "rb") as src, open(dest_dv_path, "wb") as dst:
            dst.write(src.read())
    print(f"Best model and vectorizer saved to {output_dir}/")
