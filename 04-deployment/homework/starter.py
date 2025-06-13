#!/usr/bin/env python
# coding: utf-8




# get_ipython().system('pip freeze | grep scikit-learn')


# get_ipython().system('python -V')


import pickle
import pandas as pd
import sys


# getting the year and month from the terminal arguments
year = int(sys.argv[1])
month = int(sys.argv[2])


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)




categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df



df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet')




dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


# ======================== Question 1 ========================

# standard devation of the predictions
print('Standard deviation of predictions:', y_pred.std())



# ========================= Question 2 ========================

df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

# creating the results dataframe with only ride_id and prediction
df_result = pd.DataFrame()
df_result['ride_id'] = df['ride_id']
df_result['predicted_duration'] = y_pred

output_file = "predictions.parquet"

# saving to Parquet
df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)

df_result.head(10)



import os

# getting the size in bytes
file_size_bytes = os.path.getsize(output_file)

# converting to megabytes 
file_size_mb = file_size_bytes / (1024 * 1024)

print(f"Output file size: {file_size_mb:.2f} MB")



# ========================= Question 3 ========================
print('Mean predicted duration:', y_pred.mean())



