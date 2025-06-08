from prefect import flow, task
from ingest_data import extract_tabular_data, extract_json_data
from transform_data import transform_electricity_sales_data, prepare_features, transform_capability_data
from src.model import train_model, get_best_model_uri
import os

@task
def load_data():
    sales = extract_tabular_data("data/electricity_sales.csv")
    capability = extract_json_data("data/electricity_capability_nested.json")
    return sales, capability

@task
def transform_data(sales, capability):
    sales_cleaned = transform_electricity_sales_data(sales)
    capability_cleaned = transform_capability_data(capability)
    merged_df = prepare_features(sales_cleaned, capability_cleaned)
    return merged_df

@task
def model_pipeline(df, model_type="LinearRegression"):
    model, dv, rmse, model_uri = train_model(df, model_type=model_type)
    if model_uri:
        print(f"✅ Model registered at: {model_uri}")
    else:
        print("ℹ️ No new model registered.")
    return rmse


@flow(name="Monthly Energy ML Pipeline")
def monthly_pipeline():
    sales, capability = load_data()
    merged_df = transform_data(sales, capability)
    rmse = model_pipeline(merged_df)
    print(f"Final RMSE: {rmse:.2f}")

if __name__ == "__main__":
    monthly_pipeline()
