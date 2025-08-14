# inference.py
import os
import io
import joblib
import pandas as pd

def model_fn(model_dir):
    """Load model from SageMaker model directory."""
    model_path = os.path.join(model_dir, "model.joblib")
    model = joblib.load(model_path)
    return model

def input_fn(request_body, request_content_type):
    """Parse CSV input into a DataFrame."""
    if request_content_type == "text/csv":
        # No header in CSV request: rely on training feature order
        return pd.read_csv(io.StringIO(request_body), header=None)
    # If you want to support header CSV:
    # return pd.read_csv(io.StringIO(request_body))
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Make predictions with the loaded model."""
    return model.predict(input_data)

def output_fn(prediction, content_type):
    """Return predictions in CSV."""
    if content_type == "text/csv":
        out = io.StringIO()
        pd.DataFrame(prediction).to_csv(out, index=False, header=False)
        return out.getvalue()
    raise ValueError(f"Unsupported content type: {content_type}")
