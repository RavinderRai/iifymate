import streamlit as st
import os
import time
import pickle
import logging
logging.basicConfig(level=logging.INFO)
import pandas as pd
from google.cloud import storage
from utils import priority_list_dish_type, priority_list_meal_type
from predict import preprocess_input, predict_calories, upload_artifact_to_gcs, post_process, load_artifact_from_gcs, load_pickle

def main():
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../flavourquasar-gcp-key.json"

    # Streamlit app layout
    st.title('Calorie Predictor')
    label = st.text_input("Recipe Name", "Carrot Soup")

    # can see the dish list from the keys of the skew map
    dish_type = st.selectbox("Dish Type", ['biscuits and cookies', 'bread', 'pancake', 'alcohol cocktail', 'condiments and sauces', 'desserts', 'main course', 'preps', 'preserve', 'salad', 'sandwiches', 'soup', 'starter', 'cereals', 'drinks'])
    meal_type_refined = st.selectbox("Meal Type", ['breakfast', 'lunch/dinner', 'snack'])

    user_input_raw = {
        'label': [label],
        'dishType': [dish_type],
        'mealTypeRefined': [meal_type_refined]
    }

    # Process user input and predict calories
    if st.button('Get Calories'):
        user_input = preprocess_input(user_input_raw)
        predicted_calories = predict_calories(
            user_input, num_of_classes=13, 
            artifact_path='training/XGBoost_model.pkl', 
            log_artifacts=False, 
            set_google_environment=False)
        st.write('There are roughly', predicted_calories, 'calories in {}'.format(user_input_raw['label'][0]))

if __name__ == '__main__':
    main()
