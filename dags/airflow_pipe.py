from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from tasks.movies import preprocess
from tasks.movies import train_model


dag_movies = DAG(
    dag_id="train_pipe",
    start_date=datetime(2025, 2, 1),
    max_active_tasks=4,
    schedule=timedelta(minutes=5),
    #    schedule="@hourly",
    max_active_runs=1,
    catchup=False,
)
clear_task = PythonOperator(
    python_callable=preprocess.clear_data,
    task_id="clear_movies",
    dag=dag_movies,
)
train_task = PythonOperator(
    python_callable=train_model.train,
    task_id="train_movies",
    dag=dag_movies,
)

clear_task >> train_task
