#!/bin/bash

# To run, use this command in the root directory: ml_features/ml_calorie_estimation/airflow/run_pipeline.sh

# Step 1: Sync the latest DAG
bash ml_features/ml_calorie_estimation/airflow/sync_dag.sh

# Step 2: Restart Airflow services (webserver & scheduler)
echo "♻️ Stopping any running Airflow processes..."
pkill -f "airflow webserver"  # Stop webserver if running
pkill -f "airflow scheduler"  # Stop scheduler if running

echo "🚀 Starting Airflow webserver and scheduler..."
airflow webserver &  # Start Airflow webserver in the background
airflow scheduler &  # Start Airflow scheduler in the background

# Step 3: Give Airflow a few seconds to start up
sleep 1  

# Step 4: Trigger the DAG
echo "⏳ Triggering DAG: ml_calorie_pipeline..."
airflow dags trigger ml_calorie_pipeline

echo "✅ Pipeline triggered successfully!"
