import pickle
from flask import Flask, request, jsonify


# loading the model pickle file

with open('linear_model.bin', 'rb') as file:
    (dv, model) = pickle.load(file)


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
    X = dv.transform(features)
    y_pred = model.predict(X)
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
