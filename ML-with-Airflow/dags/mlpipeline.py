from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime


# defining first task 1
def preprocess_data():
    print("Preprocessing data...")

def train_model():
    print("Training model...")

def evaluate_model():
    print("Evaluating model...")


# defining DAG
with DAG(
    "ml_pipeline",
    start_date=datetime(2025, 7, 19),
    schedule='@weekly',
) as dag:
    preprocess = PythonOperator(task_id="preprocess_task", python_callable=preprocess_data)
    train = PythonOperator(task_id="train_task", python_callable=train_model)
    evaluate = PythonOperator(task_id="evaluate_task", python_callable=evaluate_model)

    # setting the dependencies
    preprocess >> train >> evaluate
