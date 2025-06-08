## Machine Learning Pipelines
Machine learning Pipelines are a series of steps that automate the process of transforming raw data into a machine learning model. They help in managing the complexity of machine learning workflows, ensuring reproducibility, and facilitating collaboration.

## Key Components of a Machine Learning Process
1. Download the data - Ingestion
2. Transform the data - Filtering & Removing outliers
3. Preprocessing data for ML - X, y
4. Hyperparameter tuning - Best parameters
5. Train the final model 

### Machine Learning Pipeline Orchestration - Prefect

1. Start the Prefect server
   ```bash
   prefect server start
   ```

2. Ensure the MLflow backend is running
   ```bash
   mlflow server --backend-store-uri sqlite:///mlflow.db
   ```

3. Start your work pool
   ```bash
   prefect worker start --pool "taxi_duration_model_deploy"
   ```

4. Deploy your flow in the folder `ML_pipeline_orch_tools`
   ```bash
   prefect deploy ML_pipeline_prefect.py:main -n duration_prediction_model -p taxi_duration_model_deploy
   ```

5. Run the deployed flow
   ```bash
   prefect deployment run 'nyc-taxi-duration-prediction/duration_prediction_model' --param year=2021 --param month=1
   ```

