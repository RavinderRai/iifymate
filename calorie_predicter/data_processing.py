import pandas as pd
import ast
from processing_functions import round_up_to_nearest, round_down_to_nearest, filter_calories, sorted_binned_encoding, collapsing_to_priority, priority_list_dish_type, priority_list_meal_type, one_hot_encode
from sklearn.preprocessing import OneHotEncoder
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#dropping duplicates from recipe name (which is the label column) because some sources give the same recipes
raw_df = pd.read_csv('../recipes.csv')
df = raw_df.drop_duplicates('label')

def get_target_variable(df):
    #handling target variable first
    calories_df = df['calories']
    
    #capping the calorie count, so we will include recipes with calorie counts so that we maintain 90% of our data
    filtered_calories_df = filter_calories(df, column='calories', quartile_percent=0.9)
    max_calorie_cutoff = round_up_to_nearest(max(filtered_calories_df))
    
    #binning the calorie count to turn this into a classification problem
    bin_edges = [i for i in range(0, int(max_calorie_cutoff)+1, 300)]
    labels = [f"{bin_edges[i]}-{bin_edges[i+1]-1}" for i in range(len(bin_edges)-1)]
    binned_calories = pd.cut(filtered_calories_df, bins=bin_edges, labels=labels, include_lowest=True)
    binned_calories = binned_calories.rename('binnedCalories')
    
    # Assuming binned_calories is a pandas Series or DataFrame
    assert binned_calories.isna().sum() == 0, "The count of NaN values in binned_calories is not equal to 0"

    #sort and map the intervals to integers
    label_encoding = sorted_binned_encoding(binned_calories)
    target = binned_calories.map(label_encoding)

    #get the indices of the target and use them to get the same rows from the original df, then concatenate this and the target
    binned_calories_df = df.loc[target.index]
    binned_calories_df = pd.concat([binned_calories_df, target], axis=1)
    
    return binned_calories_df

def preprocess_dish_type(pre_processed_df):
    pre_processed_df = pre_processed_df.dropna(subset=['dishType'])
    dishType_df = pre_processed_df['dishType'].apply(ast.literal_eval)
    dishType_df = dishType_df.rename('dishTypeLabel')

    #turning lists of values into just singular values so this will be a categorical column
    priority_list_dish_type_var = priority_list_dish_type()
    dishType_df = dishType_df.apply(lambda x: collapsing_to_priority(x, priority_list_dish_type_var))

    # putting them together
    pre_processed_df = pd.concat([pre_processed_df, dishType_df], axis=1)

    # combining categories based on skewness, some dish types tend to have higher, lower or medium levels of calories
    skewness_by_category = pre_processed_df.groupby('dishTypeLabel')['calories'].skew()

    #separating them into 3 skewedness, so we'll take the min and max skewedness and bin all these categories
    skewness_min = skewness_by_category.min()
    skewness_max = skewness_by_category.max()
    
    interval_width = (skewness_max - skewness_min) / 3
    bin1_end = skewness_min + interval_width
    bin2_end = bin1_end + interval_width

    # just for clarity we will add a step to name these clearly, and then do label encoding right after
    bins = {
        'Left Skewed (Higher Calories)': skewness_by_category[(skewness_by_category >= skewness_min) & (skewness_by_category < bin1_end)],
        'Approximately Symmetric (Normal Calories)': skewness_by_category[(skewness_by_category >= bin1_end) & (skewness_by_category < bin2_end)],
        'Right Skewed (Lower Calories)': skewness_by_category[skewness_by_category >= bin2_end]
    }

    # flipping this around to create a map that converts each dish type label into a skewedness category
    skew_map = {}
    for skew in bins.keys():
        for category in bins[skew].index:
            skew_map[category] = skew
    
    pre_processed_df['dishTypeSkewedLabels'] = pre_processed_df['dishTypeLabel'].map(skew_map)

    #dish types with only 1 value will give nan values here, so we need to remove them
    pre_processed_df = pre_processed_df.dropna(subset=['dishTypeSkewedLabels'])
    pre_processed_df = pre_processed_df.reset_index(drop=True)

    # now we can quickly do label encoding for model training
    dish_type_map = {'Approximately Symmetric (Normal Calories)': 1, 'Right Skewed (Lower Calories)': 0, 'Left Skewed (Higher Calories)': 2}
    pre_processed_df['dishTypeSkewedLabels'] = pre_processed_df['dishTypeSkewedLabels'].map(dish_type_map)

    return pre_processed_df

def preprocess_meal_type(pre_processed_df):
    mealType_df = pre_processed_df['mealType'].apply(ast.literal_eval)

    #converting multilabel column into single label
    priority_list_meal_type_var = priority_list_meal_type()
    mealType_df = mealType_df.apply(lambda x: collapsing_to_priority(x, priority_list_meal_type_var))

    #replacing brunch and teatime with snack, effectively combining these categories
    replace_lst = ['brunch', 'teatime']
    replacement = 'snack'
    mealType_df = mealType_df.apply(lambda x: replacement if x in replace_lst else x)

    mealType_df = mealType_df.rename('mealTypeRefined')
    pre_processed_df = pd.concat([pre_processed_df, mealType_df], axis=1)

    return pre_processed_df

english_stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()




pre_processed_df = get_target_variable(df)
pre_processed_df = preprocess_dish_type(pre_processed_df)
pre_processed_df = preprocess_meal_type(pre_processed_df)
onehot_encoded_df = one_hot_encode(pre_processed_df, 'mealTypeRefined')
pre_processed_df = pd.concat([pre_processed_df, onehot_encoded_df], axis=1)

print(pre_processed_df.shape)