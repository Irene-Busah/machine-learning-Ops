# pipeline.py
import os
import boto3
import sagemaker
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import CSVDeserializer
from time import gmtime, strftime

import pandas as pd
from sklearn.model_selection import train_test_split

# ====== Configure these ======
ROLE_ARN = "arn:aws:iam::314146298520:role/sagemaker-ml"   # <- put your role here
BUCKET   = "mob-price-sagemaker15"                       # <- put your bucket here
PREFIX   = "sagemaker/mobile_price_classification/sklearn"    # S3 key prefix
REGION   = boto3.Session().region_name

# ====== Local data prep ======
LOCAL_DATA = "data/mob_price_classification_train.csv"
assert os.path.exists(LOCAL_DATA), f"Missing {LOCAL_DATA}"

df = pd.read_csv(LOCAL_DATA)

# Ensure label exists
assert "price_range" in df.columns, "Expected label column 'price_range'"

X = df.drop(columns=["price_range"])
y = df["price_range"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

train_df = X_train.copy()
train_df["price_range"] = y_train.values
test_df = X_test.copy()
test_df["price_range"] = y_test.values

# Write out split files
TRAIN_FILE = "train-v1.csv"
TEST_FILE  = "test-v1.csv"
train_df.to_csv(TRAIN_FILE, index=False)
test_df.to_csv(TEST_FILE, index=False)

# ====== Upload to S3 ======
session = sagemaker.Session()
s3_train = session.upload_data(path=TRAIN_FILE, bucket=BUCKET, key_prefix=PREFIX)
s3_test  = session.upload_data(path=TEST_FILE,  bucket=BUCKET, key_prefix=PREFIX)

print("S3 train path:", s3_train)
print("S3 test path :", s3_test)

# ====== Configure Estimator (training) ======
# Using the classic SKLearn container 0.23-1 (matches your previous code)
FRAMEWORK_VERSION = "0.23-1"

estimator = SKLearn(
    entry_point="train.py",                 # our training script below
    role=ROLE_ARN,
    instance_count=1,
    instance_type="ml.m5.large",
    framework_version=FRAMEWORK_VERSION,
    base_job_name="rf-mobile-price",
    hyperparameters={
        "n_estimators": 200,
        "random_state": 42
    },
    # Optional: use spot and cap runtime
    use_spot_instance=True,
    max_run=3600,
)

# Launch training job with named channels 'train' and 'test'
estimator.fit({"train": s3_train, "test": s3_test}, wait=True)

# ====== Create Model object for deployment (inference) ======
model_name = "rf-mobile-price-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
sk_model = SKLearnModel(
    name=model_name,
    model_data=estimator.model_data,   # S3 path to model.tar.gz produced by training
    role=ROLE_ARN,
    entry_point="inference.py",        # our inference script below
    framework_version=FRAMEWORK_VERSION,
)

# ====== Deploy endpoint ======
endpoint_name = "rf-mobile-price-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
print("EndpointName:", endpoint_name)

predictor = sk_model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name=endpoint_name
)

# Set CSV I/O
predictor.serializer   = CSVSerializer()
predictor.deserializer = CSVDeserializer()

# ====== Test a prediction (take a few rows from our test set) ======
sample = X_test.head(5)
print("Sending sample for prediction:")
print(sample)

preds = predictor.predict(sample.to_csv(index=False, header=False))
print("Predictions:", preds)

print(f"\n✅ Deployed endpoint: {endpoint_name}")
print("Tip: run predictor.delete_endpoint() when done.")
