from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import subprocess
from airflow.models import Variable

"""
If you make a change to this file, make sure to run:
cp ml_features/ml_calorie_estimation/airflow/dags/ml_pipeline.py ~/airflow/dags/
to update the DAG in Airflow
"""

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2025, 3, 9),
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

dag = DAG(
    "ml_calorie_pipeline",
    default_args = default_args,
    description="ML pipeline for calorie estimation",
    schedule_interval=None, # Run ML pipeline manually
    catchup=False
)

env = Variable.get("ENV", default_var="local")

def run_script(script_name):
    """Runs a script using subprocess and ensures Airflow environment variables are passed."""
    script_path = f"ml_features.ml_calorie_estimation.pipeline.{script_name}"
    env_vars = os.environ.copy()  # Copy current environment variables
    env_vars["ENV"] = Variable.get("ENV", default_var="local")  # Set ENV variable
    
    #subprocess.run(["/home/ravib/projects/iifymate/.iifymate/bin/python", "-m", script_path], check=True, env=env_vars)

    command = f"source /home/ravib/projects/iifymate/.iifymate/bin/activate && python -m {script_path}"

    subprocess.run(
        command,
        shell=True,  # Required for `source` to work
        executable="/bin/bash",  # Ensure correct shell is used
        check=True,
        env=env_vars
    )
    
# Define tasks
data_ingestion_task = PythonOperator(
    task_id="data_ingestion",
    python_callable=run_script,
    op_args=["data_ingestion"],
    dag=dag,
)

data_cleaning_task = PythonOperator(
    task_id="data_cleaning",
    python_callable=run_script,
    op_args=["data_cleaning"],
    dag=dag,
)

feature_engineering_task = PythonOperator(
    task_id="feature_engineering",
    python_callable=run_script,
    op_args=["feature_engineering"],
    dag=dag,
)

train_task = PythonOperator(
    task_id="train_model",
    python_callable=run_script,
    op_args=["train"],
    dag=dag,
)

# Define dependencies (sequential execution)
data_ingestion_task >> data_cleaning_task >> feature_engineering_task >> train_task
