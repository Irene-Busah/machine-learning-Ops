"""
load_data.py

This module provides functionality for loading datasets required for Parkinson's disease prediction.
It is part of the `parkinson_disease_prediction` project.

The primary purpose of this file is to define functions that handle the loading of data from various
file formats, ensuring that the data is ready for further analysis or modeling.

Functions:
-----------
- load_dataset(filepath): Loads a dataset from a specified file path. Supports CSV and Parquet formats.

Usage:
-------
This module is intended to be imported and used by other scripts or modules within the project.
Example:
    data = load_dataset("path/to/dataset.csv")

Notes:
-------
- Ensure that the dataset file paths are correctly specified before using the functions in this module.
- Add any additional helper functions as needed to support data loading and preprocessing.

Raises:
-------
- FileNotFoundError: If the specified file does not exist.
- ValueError: If the file format is not supported or the data cannot be loaded.
"""

# importing necessary libraries
import pandas as pd
import os


def load_dataset(filepath) -> pd.DataFrame:
    """
    Loads the dataset from the specified file path.

    Args:
        filepath (str): The path to the file containing the dataset.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the loaded dataset.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file format is not supported or the data cannot be loaded.
    """

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    try:
        if filepath.endswith(".csv"):
            return pd.read_csv(filepath)
        elif filepath.endswith(".parquet"):
            return pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
    
    except Exception as e:
        raise ValueError(f"Failed to load data from {filepath}")

