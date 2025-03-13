#!/bin/bash

# Sync only the ml_pipeline.py DAG file
cp ml_features/ml_calorie_estimation/airflow/dags/ml_pipeline.py ~/airflow/dags/
echo "✅ DAG file updated in Airflow directory."
