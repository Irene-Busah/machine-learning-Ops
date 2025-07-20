"""
taskflow.py
=============

Orchestrating pipeline tasks using Taskflow API
"""


# importing the necessary libraries
from datetime import datetime
from airflow.decorators import task
from airflow import DAG


with DAG('taskflow-arithmetic-schedule', start_date=datetime(2025, 7, 20), schedule=('@once'), catchup=False) as dag:

    # defining the tasks

    @task
    def start_value():
        initial_value = 10
        print(f"Initial Starting value: {initial_value}")
        return initial_value
    
    @task
    def add_five(number):
        new_value = number + 5
        print(f"Added 5 to the number: {new_value}")
        return new_value
    
    @task
    def multiply_by_two(number):
        new_value = number * 2
        print(f"Multiplied by 2 and the new number: {new_value}")
        return new_value
    
    @task
    def subtract_three(number):
        new_value = number - 3
        print(f"Subtracted 3 and the New Value: {new_value}")
        return new_value
    
    @task
    def square_number(number):
        new_value = number ** 2
        print(f"Squared the number and the New Value: {new_value}")
        return new_value
    

    # setting up dependencies
    start_number = start_value()
    add_value = add_five(start_number)
    multiply_value = multiply_by_two(add_value)
    subtract_value = subtract_three(multiply_value)
    square_value = square_number(subtract_value)

