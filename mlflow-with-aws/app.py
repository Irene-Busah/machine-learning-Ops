# importing necessary libraries
import os
import sys


import mlflow.sklearn
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse

import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


os.environ['MLFLOW_TRACKING_URI'] = "http://ec2-44-202-86-198.compute-1.amazonaws.com:5000"



# function to evaluate the model
def evaluate_metric(actual, pred):
    rmse = root_mean_squared_error(actual, pred)
    mae = mean_absolute_error(actual, pred)
    r_sqaured = r2_score(actual, pred)

    return rmse, mae, r_sqaured

if __name__ == "__main__":
    # loading the data
    data_path = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-white.csv"

    try:
        data = pd.read_csv(data_path, sep=";")
    except Exception as e:
        logger.exception("Unable to load dataset")
    
    # splitting the dataset
    x = data.drop(columns=['quality'])
    y = data['quality']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run(run_name="wine-quality"):

        # building the model
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)

        # fitting the model
        model.fit(x_train, y_train)

        # predicting the model
        pred = model.predict(x_test)

        # evaluating the model
        (rmse, mae, r_square) = evaluate_metric(y_test, pred)

        print(f"ElasticNet Model (alpha={alpha}, l1_ratio={l1_ratio})")
        print(f"RMSE: {rmse}")
        print(f"MAE: {mae}")
        print(f"R2-Squared: {r_square}")

        # logging the parameters & metrics
        mlflow.log_param("Alpha", alpha)
        mlflow.log_param("L1_ratio", l1_ratio)
        
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2-Sqaured", r_square)

        signature = infer_signature(y_test, pred)

        # setting the remote server
        remote_server_uri = "http://ec2-44-202-86-198.compute-1.amazonaws.com:5000"
        mlflow.set_tracking_uri(remote_server_uri)

        tracking_url_type = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type != "file":
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="models",
                signature=signature,
                registered_model_name="Best Model - ElasticNet"
            )
        else:
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="models",
                signature=signature
            )


