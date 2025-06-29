# MLFlow with AWS
In this project, we will tracking a machine learning experiments using MLFlow hosted on AWS

### MLFLOW on AWS
```bash
mlflow server -h 0.0.0.0 --default-artifact-root s3://{S3_BUCKET_NAME}
mlflow server -h 0.0.0.0 --default-artifact-root s3://wine-quality-mlflow-model
```


