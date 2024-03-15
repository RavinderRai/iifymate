# FlavourQuasar App - Recipe Calorie Prediction

## Introduction

The aim of this project is to experiment with an easier way to count calories using machine learning. It is a common issue where people don't want to go through the effort to count calories, macro's, etc., but with machine learning we can fix that by making it easier by outputting this information given just the recipe name. For now, the calories are outputted as ranges of 300, as the goal is to just provide users with an estimate of the calorie content of a recipe before preparing it. 

[Live Link](https://streamlit-cp-go4e5g7vpq-uc.a.run.app)

<img src="flavourquasar_coverphoto.jpeg" alt="FlavourQuasar Cover Photo" width="500" height="350">

## Data Collection

The dataset used for this project consists of a collection of recipes, taken from the Edamam API, each labeled with its name, dish type (e.g., appetizer, main course, dessert), meal type (e.g., breakfast, lunch/dinner, snack), and the corresponding calorie content. This Edamam API comes with free and paid versions, where the limited free version was used. Unfortuneately this means arbitrary searches can't be made to get a large assortment of random recipes, so instead we searched with single item ingredients, and pulled 100 recipes. Thus, with a list of single ingredients, we pulled 100 recipes for each and so our dataset has 100 times the number of ingredients we use, so more can always be added.

## EDA and Preprocessing

The final input variables in the end were the recipe name, with column name label, the dish type, and the meal type. Some recipes were duplicates based on the recipe name, due to the fact that they originated from different websites but were the same otherwise - and so were dropped. Otherwise the dataset was overall clean. 

The meal type and dish type variable both come as multilabel datasets and so were collapsed into single/categorical variables using a priority list. Then, the dish type variable was split into 3 categories based on skewedness, since it had so many in it's raw form. Since the skewedness was obvious and apparent here, and there was sort of an order to it, this variable was label encoded for model training. Next, the meal type only had 5 categories and 3 were minority categories and somewhat similar, so these were combine into one, resulting in a 3 category column as well. But, the skewedness was not apparent here so this variable was one hot encoded instead. The last input variable was the recipe name, which was a text column. Thus, standard preprocessing steps, like removing stop words, lemmatization, etc., were done, right before tf-idf was applied for model training. 

**Meal Type Calorie Distribution:**

<img src="flavourquasar_calorie_distribution_meal_type.jpg" alt="MealType Calorie Distribution Image" width="1000" height="270">

Finally, the original target variable, being a continuous value for the calorie count, had a large range of values that was not intuitive. You can see the plot below of its distribution, but recipes being more than 2000 calories for 1 person is unusual so they must've been large meals for multiple people and thus were removed. To keep most of the data though, we made sure to only cut out data so that 90% was kept. Then, the calorie counts were modified to be binned into calorie ranges of 300 to make this a classification problem. 

<img src="flavourquasar_calorie_distribution.jpg" alt="Calorie Distribution Image" width="500" height="300">

## Modeling

Various models were fitted but XGBoost with tf-idf pre-processing seems to be thes best so far. The trained model is evaluated using the cohen kappa score, which is a metric meant for multi-classification problems so works well here. Although it is a harsh metric as well, performance is still low at the moment so will continue to update, and tracking will be done with either mlflow or GCP. Furthermore, grid search is used to get the best model and parameters are tracked as well.

## Deployment and Future Improvements
The app is deployed into a streamlit app for now as a test, but as features get added this can become a full micro-SaaS potentially, and would be best suited for mobile. Future improvements include building out models to predict macro nutrients, not just calories. Also, an LLM could be trained to predict ingredients or the full recipe to make the meal, and experimentation has already started so in the future could finish this. Potentially could make the app predict 5 different recipe versions, and the user could pick the most relevant one, and then another model would predict calories/macros from that - increasing accuracy.
