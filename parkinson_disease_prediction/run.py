"""
run.py

This script is responsible for executing the machine learning model for Parkinson's disease prediction.
It serves as the entry point for loading the model, processing input data, and generating predictions.

Usage:
    Execute this script to run the prediction pipeline for Parkinson's disease.

Modules:
    - Ensure all required dependencies and modules are installed before running this script.

Author:
    [Your Name or Organization]

Date:
    [Date of Creation or Last Modification]
"""


# importing the defined functions
from src.data.load_data import load_dataset


# importing libraries
import pandas as pd
import os


if __name__ == "__main__":
    filepath = "data/parkinson_disease.csv"
    # print(os.path.isfile(filepath))

    data = load_dataset(filepath=filepath)
    
    print(data.shape)

    print(data.head(10))
