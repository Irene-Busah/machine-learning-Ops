"""
nasa_etl_pipeline.py
=======================

Utilizes Apache Airflow to orchestrate the ETL pipeline process
"""


# importing the necessary libraries
from airflow import DAG
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.decorators import task
from airflow.providers.postgres.hooks.postgres import PostgresHook
from datetime import datetime, timedelta
import json


# defining the DAG
with DAG(
    dag_id='nasa_apod_postgres',
    start_date=datetime.now() - timedelta(days=1),
    schedule='@daily',
    catchup=False
) as dag:
    
    # creating the table if it doesn't exists
    @task
    def create_table():
        """Create a table, if it doesn't exist"""
        postgres_hook = PostgresHook(postgres_conn_id='postgres_connection')

        create_table_query = """
        CREATE TABLE IF NOT EXISTS apod_data (
            id SERIAL PRIMARY KEY,
            title VARCHAR(255),
            explanation TEXT,
            url TEXT,
            date DATE,
            media_type VARCHAR(50)
        );
        """
        # executing the table creation
        postgres_hook.run(create_table_query)
    
    # https://api.nasa.gov/
    extract_data=SimpleHttpOperator(
        task_id='extract_data_task',
        http_conn_id='nasa_api',     # Connection ID defined in Airflow for NASA API
        endpoint='planetary/apod',   # NASA API endpoint for APOD
        method='GET',
        data={"api_key":"{{ conn.nasa_api.extra_dejson.api_key }}"},  # Use the API Key from the connection
        response_filter=lambda response:response.json()
    )

    # transforming task
    @task
    def transform_data_data(response):
        data = {
            'title': response.get('title', ''),
            'explanation': response.get('explanation', ''),
            'url': response.get('url', ''),
            'date': response.get('date', ''),
            'media_type': response.get('media_type', '')
        }

        return data


    # loading task
    @task
    def load_data_to_postgres(data):
        postgres_hook = PostgresHook(postgres_conn_id='postgres_connection')

        # inserting the data into the postgres table
        insert_query = """
        INSERT INTO apod_data (title, explanation, url, date, media_type)
        VALUES (%s, %s, %s, %s, %s)
        """

        postgres_hook.run(insert_query, parameters=(
            data['title'],
            data['explanation'],
            data['url'],
            data['date'],
            data['media_type']
        ))



# defining the dependencies
create_table() >> extract_data
api_response = extract_data.output
transform_data = transform_data_data(api_response)
load_data_to_postgres(transform_data)
