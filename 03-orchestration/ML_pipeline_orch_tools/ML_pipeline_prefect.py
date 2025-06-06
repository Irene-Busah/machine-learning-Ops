#!/usr/bin/env python
# coding: utf-8

# mlflow server --backend-store-uri sqlite:///mlflow.db

# running the pipeline: python .\duration-prediction.py --year=2021 --month=1

import pandas as pd
import pickle
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
from prefect import task, flow
import mlflow


# mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-experiment")


@task(name="load-data", description="loads data", retries=3, retry_delay_seconds=5)
def load_data(year, month):
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet"

    df = pd.read_parquet(url)


    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    
    return df


@task
def create_X(df, dv=None):
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv


# defining a function to train the model and log the results
@task(log_prints=True)
def train_and_log_model(X_train, y_train, X_val, y_val, dv):
    mlflow.xgboost.autolog(disable=True)

    with mlflow.start_run():
        
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        best_params = {
            'learning_rate': 0.09585355369315604,
            'max_depth': 30,
            'min_child_weight': 1.060597050922164,
            'objective': 'reg:linear',
            'reg_alpha': 0.018060244040060163,
            'reg_lambda': 0.011658731377413597,
            'seed': 42
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=30,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )

        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")


@flow(name="nyc-taxi-duration-prediction", description="Flow to predict NYC taxi trip duration")
def main(year, month):
    df_train = load_data(year, month)

    next_month = month + 1 if month < 12 else 1
    next_year = year if month < 12 else year + 1
    df_val = load_data(next_year, next_month)

    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv=dv)

    y_train = df_train['duration'].values
    y_val = df_val['duration'].values

    train_and_log_model(X_train, y_train, X_val, y_val, dv)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a model to predict taxi trip duration.")
    parser.add_argument("--year", type=int, required=True, help="Year of the training data")
    parser.add_argument("--month", type=int, required=True, help="Month of the training data")
    args = parser.parse_args()

    main(args.year, args.month)