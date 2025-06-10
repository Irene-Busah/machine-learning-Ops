import requests


ride = {
    "PULocationID": 1,
    "DOLocationID": 2,
    "trip_distance": 3.5
}

response = requests.post(url='http://0.0.0.0:9696/predict', json=ride).json()
print(response)

