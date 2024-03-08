# FlavourQuasar App

[Live Link](https://streamlit-cp-go4e5g7vpq-uc.a.run.app)

# Recipe Calorie Prediction

Overview
Predicting the calorie content of recipes based on primarily the recipe name. The calories are outputted as ranges of 300, as the goal is to just provide users with an estimate of the calorie content of a recipe before preparing it.

Dataset
The dataset used for this project consists of a collection of recipes, taken from the Edamam API, each labeled with its name, dish type (e.g., appetizer, main course, dessert), meal type (e.g., breakfast, lunch/dinner, snack), and the corresponding calorie content. The calorie values are binned into groups of 300 to simplify the prediction task.

Approach
Data Preprocessing: The dataset is cleaned and preprocessed to handle missing values, encode categorical variables, and prepare features for model training. The meal type and dish type variable both come as multilabel datasets and are collapsed into single/categorical variables. The recipe name is preprocessed in the standard manner for tf-idf pre-processing.

Feature Engineering: The original target variable, being a continuous value for the calorie count, was modified to be binned into calorie ranges of 300.This is to turn the problem into a classification problem, since a fair amount of data is lost from removing outliers.  

Model Selection: Various models were fitted but XGBoost with tf-idf pre-processing seems to be thes best so far, though performance is still low at the moment so will continue to update and track with mlflow or GCP.

Model Training: Grid search is used to get the best model.

Model Evaluation: The trained model is evaluated using the cohen kappa score, which is a metric meant for multi-classification problems so works well here. 

Deployment: The app is deployed into a streamlit app for now, but as features get added this can be moved to mobile.
