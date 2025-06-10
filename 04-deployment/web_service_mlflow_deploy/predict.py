import pickle
from flask import Flask, request, jsonify
import mlflow

from mlflow.tracking import MlflowClient



RUN_ID = "1f93ff6e9637480d875387c63b33b847"
MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000'

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

print("Setting Tracking URI...")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


logged_model = f'runs:/{RUN_ID}/model'

# Load model as a PyFuncModel.
model = mlflow.pyfunc.load_model(logged_model)


def get_features(ride):
    """
    Extract features from the ride data.
    :param ride: A dictionary containing ride data.
    """

    features = {}

    features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']

    return features

def predict(features):
    """
    Predict the target value using the loaded model and features.

    :param features: A dictionary containing the input features.
    :return: The predicted target value.
    """
    # X = model[features]
    y_pred = model.predict(features)
    return y_pred[0]


# create the Flask app
app = Flask('predict')

@app.route('/predict', methods=['POST'])
def predict_ride():
    """
    Handle the prediction request.
    :return: JSON response with the predicted fare amount.
    """
    ride = request.get_json()
    features = get_features(ride)
    y_pred = predict(features)
    result = {'duration': float(y_pred)}
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9696, debug=True)
