## End-to-End Sagemaker Project

What is SageMaker?

Amazon SageMaker is AWS’s managed service for the entire ML lifecycle:

- Data Preparation → store, label, and process data in S3 or SageMaker Data Wrangler.
- Model Training → train on AWS-managed compute (no local GPU needed).
- Model Deployment → create real-time APIs or batch inference jobs.
- Monitoring → check endpoint performance, retrain when needed.
- MLOps → automate with SageMaker Pipelines.

It supports both built-in algorithms and custom code (like the train.py and inference.py we just wrote).



2️⃣ Core Concepts
Before you write a line of code, remember these building blocks:

| Concept       | What it is                                    | Example in our project                                     |
| ------------- | --------------------------------------------- | ---------------------------------------------------------- |
| **S3 Bucket** | Storage for data & models                     | `s3://mybucket/prefix/train-v1.csv`                        |
| **IAM Role**  | Permissions for SageMaker to access S3        | `arn:aws:iam::1234567890:role/sagemaker-ml`                |
| **Estimator** | Object that launches training jobs            | `SKLearn(entry_point="train.py", ...)`                     |
| **Channels**  | Named input data sources                      | `"train": s3_train, "test": s3_test`                       |
| **Model**     | A packaged artifact + inference script        | `SKLearnModel(model_data=..., entry_point="inference.py")` |
| **Endpoint**  | Live API that serves predictions              | `predictor = model.deploy(...)`                            |
| **Predictor** | Python object to send requests to an endpoint | `predictor.predict(sample)`                                |



3️⃣ The General Workflow
Think of SageMaker as a factory:

1. 📦 Get the materials → put data in S3
2. 🏗️ Build the model → run training job in SageMaker
3. 🚚 Package the model → store model artifacts in S3
4. 🏭 Deploy the model → run in an endpoint
5. 🧪 Test the model → send predictions from your code

