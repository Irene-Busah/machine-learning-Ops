"""
DAG Task as follows

1. Start with an initial number
2. Add 5 to the number
3. Multiply the result by 2
4. Subtract 3 from the result
5. Compute the square of the result
"""


# importing relevant libraries
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime


# defining the function for each week
def start_number(**context):
    context['ti'].xcom_push(key='current_value', value=10)
    print("Starting number 10")

def add_five(**context):
    current_value=context['ti'].xcom_pull(key='current_value', task_ids='start_task')
    new_value = current_value + 5
    context['ti'].xcom_push(key='current_value', value=new_value)
    print(f"Added 5 and the New Value is {new_value}")

def multiply_by_two(**context):
    current_value=context['ti'].xcom_pull(key='current_value', task_ids='add_task')
    new_value = current_value * 2
    context['ti'].xcom_push(key='current_value', value=new_value)
    print(f"Multiply by 2 and the New Value is {new_value}")

def subtract_three(**context):
    current_value=context['ti'].xcom_pull(key='current_value', task_ids='multiply_task')
    new_value = current_value - 3
    context['ti'].xcom_push(key='current_value', value=new_value)
    print(f"Subtract 3 and the New Value is {new_value}")

def square_value(**context):
    current_value=context['ti'].xcom_pull(key='current_value', task_ids='subtract_task')
    new_value = current_value ** 2
    # context['value'].xcom_push(key='current_value', value=new_value)
    print(f"Squared the value and the New Value is {new_value}")


with DAG(
    'arithmetic_operation',
    start_date=datetime(2025, 7, 19), schedule='@once',
    catchup=False
) as dag:
    
    start_task = PythonOperator(
        task_id='start_task',
        python_callable=start_number
    )

    add_task = PythonOperator(
        task_id='add_task',
        python_callable=add_five
    )

    multiply_task = PythonOperator(
        task_id='multiply_task',
        python_callable=multiply_by_two
    )

    subtract_task = PythonOperator(
        task_id='subtract_task',
        python_callable=subtract_three
    )

    square_task = PythonOperator(
        task_id='square_task',
        python_callable=square_value
    )

    start_task >> add_task >> multiply_task >> subtract_task >> square_task

    