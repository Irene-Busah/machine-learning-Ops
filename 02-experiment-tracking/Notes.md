# Experiment Tracking - MLOps

- Machine Learning Experiment is the process of building an ML model
- Experiment Run refers to each trial in an ML experiment
- Run artifact refers to any file that is associated with an ML Run


Experiment tracking is the process of keeping track of all the relevant information from an ML experiment, which includes source code, environment, data, model, hyper-parameters and metrics.

### MLflow Modules
1. Tracking
This module allows you to organize experiments into runs and to keep track of parameters, metrics, metadata, artifacts & models

2. Models
3. Model Registry
4. Projects


### Configuring MLflow
1. Backend Store
- SQLAlchemy like SQLite
- Local file system
- Remote file system

2. Artifact Store
- Local file system
- Remote file system like S3, GCS, Azure Blob Storage

3. Tracking Server
- No tracking server
- Local tracking server like localhost
- Remote tracking server like AWS, GCP, Azure
