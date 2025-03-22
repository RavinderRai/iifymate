#!/bin/bash

# To run, use this command in the root directory: 
# bash ml_features/ml_calorie_estimation/airflow/run_pipeline_step.sh

# Step 1: Sync the latest DAG
bash ml_features/ml_calorie_estimation/airflow/sync_dag.sh

# Step 2: Restart Airflow services (webserver & scheduler)
echo "♻️ Stopping any running Airflow processes..."
pkill -f "airflow webserver"  # Stop webserver if running
pkill -f "airflow scheduler"  # Stop scheduler if running

echo "🚀 Starting Airflow webserver and scheduler..."
nohup airflow webserver > ~/airflow/webserver.log 2>&1 &
nohup airflow scheduler > ~/airflow/scheduler.log 2>&1 &

# Step 3: Give Airflow a few seconds to start up
sleep 3  

# Step 4: List Available Pipeline Steps
echo "📌 Select a pipeline step to run:"
options=("data_ingestion" "data_cleaning" "feature_engineering" "train_model" "deploy_model" "Exit")
select step in "${options[@]}"
do
    case $step in
        "data_ingestion")
            echo "⏳ Running Data Ingestion..."
            airflow tasks run ml_calorie_pipeline data_ingestion manual__$(date -u +"%Y-%m-%dT%H:%M:%S") --local
            break
            ;;
        "data_cleaning")
            echo "⏳ Running Data Cleaning..."
            airflow tasks run ml_calorie_pipeline data_cleaning manual__$(date -u +"%Y-%m-%dT%H:%M:%S") --local
            break
            ;;
        "feature_engineering")
            echo "⏳ Running Feature Engineering..."
            airflow tasks run ml_calorie_pipeline feature_engineering manual__$(date -u +"%Y-%m-%dT%H:%M:%S") --local
            break
            ;;
        "train_model")
            echo "⏳ Running Model Training..."
            airflow tasks run ml_calorie_pipeline train_model manual__$(date -u +"%Y-%m-%dT%H:%M:%S") --local
            break
            ;;
        "deploy_model")
            echo "⏳ Running Model Deployment..."
            airflow tasks run ml_calorie_pipeline deploy_model manual__$(date -u +"%Y-%m-%dT%H:%M:%S") --local
            break
            ;;
        "Exit")
            echo "❌ Exiting script."
            exit 0
            ;;
        *)
            echo "🚨 Invalid option! Please select a valid pipeline step."
            ;;
    esac
done

echo "✅ Pipeline step '$step' executed successfully!"
