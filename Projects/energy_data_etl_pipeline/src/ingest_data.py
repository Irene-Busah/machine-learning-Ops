"""
ingest.py
----------

This module handles data ingestion for the energy pipeline project.

It is responsible for reading raw datasets from different sources:
1. `electricity_sales.csv`: contains electricity sales data with prices by state and sector
2. `electricity_capability_nested.json`: contains nested capability data for energy types

The script performs basic data ingestion task from different sources for use by downstream pipeline stages.

Expected Input:
- CSV: data/electricity_sales.csv
- JSON: data/electricity_capability_nested.json

Output:
- Pandas DataFrames for each dataset
"""


# importing necessary libraries
import pandas as pd
import json


def extract_tabular_data(file_path: str):
    """Extract data from a tabular file_format, with pandas."""
    try:
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
        elif file_path.endswith(".parquet"):
            data = pd.read_parquet(file_path)
    except Exception as e:
        raise Exception("Warning: Invalid file extension. Please try with .csv or .parquet!")
    return data



def extract_json_data(file_path):
    """Extract and flatten data from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            data = pd.json_normalize(data)
    except Exception as e:
        raise Exception("Warning: Invalid file extension. Please try with .json!")
    return data
    