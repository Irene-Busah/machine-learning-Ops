"""
transform.py
------------

This module handles data transformation for the energy pipeline.

It loads the cleaned outputs from `ingest.py`, applies transformation logic,
and produces a modeling-ready dataset by joining electricity sales and capability data.

Expected Inputs:
- data/processed/sales_clean.csv
- data/processed/capability_clean.csv

Outputs:
- data/processed/merged_data.csv

Usage:
    Run as part of pipeline or test standalone for debugging feature transformations.
"""


# importing necessary libraries
import pandas as pd
import os

# creating output directory if it does not exist
os.makedirs('data/processed', exist_ok=True)


def transform_electricity_sales_data(raw_data: pd.DataFrame):
    """
    Transform electricity sales to find the total amount of electricity sold
    in the residential and transportation sectors.
    """

    # 1. dropping the NA values from the `price` column
    raw_data.dropna(subset=['price'], inplace=True)

    # 2. creating a new column `year` using the first 4 characters of `period`
    raw_data['year'] = raw_data['period'].str[:4]

    # 3. creating a new column `month` using the last 2 characters of `period`
    raw_data['month'] = raw_data['period'].str[-2:]

    # 5. returning the transformed DataFrame with only the required columns
    return raw_data[['year', 'month', 'stateid', 'sectorName', 'price', 'price-units']]


def transform_capability_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and restructure flattened capability data from the JSON file.

    Args:
        raw_data (pd.DataFrame): Raw capability data after JSON flattening

    Returns:
        pd.DataFrame: Cleaned DataFrame with standardized column names and types
    """
    # Rename columns for consistency
    df = raw_data.rename(columns={
        "stateId": "stateid",
        "energySource.capability": "capability",
        "energySource.capabilityUnits": "capability_units"
    })

    # Convert capability to numeric (it may be string)
    df["capability"] = pd.to_numeric(df["capability"], errors="coerce")

    # Ensure period is treated as a string to extract the year easily
    df["period"] = df["period"].astype(str)

    # Keep only required columns
    return df[["stateid", "period", "capability", "capability_units"]].dropna()


def prepare_features(sales_df: pd.DataFrame, capability_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join and transform sales + capability data into a modeling-ready dataset.

    Args:
        sales_df (pd.DataFrame): Clean electricity sales data
        capability_df (pd.DataFrame): Flattened capability data

    Returns:
        pd.DataFrame: Feature-enhanced merged dataset
    """
    sales_df["year"] = pd.to_datetime(sales_df["year"], errors="coerce").dt.year
    capability_df["year"] = pd.to_numeric(capability_df["period"], errors="coerce")

    merged_df = pd.merge(
    sales_df,
    capability_df,
    how="inner",
    left_on=["stateid", "year"],
    right_on=["stateid", "year"]
)

    merged_df["price_per_mw"] = merged_df["price"] / (merged_df["capability"] + 1e-5)
    return merged_df.dropna(subset=["price", "capability"])


def save_transformed_data(df: pd.DataFrame):
    df.to_csv("data/processed/merged_data.csv", index=False)
    print("Transformed data saved to `data/processed/merged_data.csv`")
