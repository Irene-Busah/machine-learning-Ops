#!/usr/bin/env python
# coding: utf-8


# importing the necessary libraries
import os
import uuid
import pickle

import mlflow

import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import make_pipeline



mlflow.set_tracking_uri("http://127.0.0.1:5000")

# mlflow.set_experiment("green-taxi-experiment")


year = 2021
month = 2
taxi_type = "yellow"


input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
output_file = f'output/{taxi_type}/{year:04d}-{month:02d}.parquet'

RUN_ID = '8fd3dc15a36d4240bf05b54b75ed1541'



def generate_uuids(n):
    """Generate a list of n unique UUIDs."""

    ride_ids = []
    for i in range(n):
        ride_ids.append(str(uuid.uuid4()))
    return ride_ids



def read_dataframe(filename):
    """Read a DataFrame from a Parquet file."""

    data = pd.read_parquet(filename)

    if taxi_type == 'green':
        pickup_col = 'lpep_pickup_datetime'
        dropoff_col = 'lpep_dropoff_datetime'
    else:
        pickup_col = 'tpep_pickup_datetime'
        dropoff_col = 'tpep_dropoff_datetime'

    data['duration'] = data[dropoff_col] - data[pickup_col]
    data['duration'] = data['duration'].dt.total_seconds() / 60

    data = data[(data['duration'] >= 1) & (data['duration'] <= 60)]
    data['ride_id'] = generate_uuids(len(data))

    return data



def prepare_dictionaries(data):
    """Prepare a dictionary representation of the DataFrame for feature extraction."""

    categorical = ['PULocationID', 'DOLocationID']
    data[categorical] = data[categorical].astype(str)

    data['PU_DO'] = data['PULocationID'] + '_' + data['DOLocationID']

    categorical = ['PU_DO']
    numerical = ['trip_distance']

    dicts = data[categorical + numerical].to_dict(orient='records')
    return dicts



def load_model(run_id):
    logged_model = f'runs:/{run_id}/model'
    model = mlflow.pyfunc.load_model(logged_model)
    return model


def apply_model(input_file, run_id, output_file):
    """Apply the model to the input data and save the results to a Parquet file."""

    print(f"Applying model with run_id: {run_id} to input file: {input_file}")
    df = read_dataframe(input_file)
    dicts = prepare_dictionaries(df)

    print("Applying model to the data...")
    model = load_model(run_id)
    y_pred = model.predict(dicts)

    print("Model applied successfully.")
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']

    # Dynamically choose pickup datetime column based on taxi type
    pickup_col = 'lpep_pickup_datetime' if taxi_type == 'green' else 'tpep_pickup_datetime'
    df_result['pickup_datetime'] = df[pickup_col]

    df_result['PULocationID'] = df['PULocationID']
    df_result['DOLocationID'] = df['DOLocationID']
    df_result['actual_duration'] = df['duration']
    df_result['predicted_duration'] = y_pred
    df_result['diff'] = df_result['actual_duration'] - df_result['predicted_duration']
    df_result['model_version'] = run_id

    print("Saving results to Parquet file...")
    df_result.to_parquet(output_file, index=False)
    return df_result




print(apply_model(input_file=input_file, run_id=RUN_ID, output_file=output_file))

