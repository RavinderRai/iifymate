import pandas as pd
import warnings
from sklearn.preprocessing import OneHotEncoder

def round_up_to_nearest(number, bin_size=300):
    # Round up to the nearest multiple of 100
    rounded_number = ((number + 99) // 100) * 100
    # Find the nearest multiple of 300 that is greater than or equal to the rounded number
    rounded_number = ((rounded_number + (bin_size-1)) // bin_size) * bin_size
    return rounded_number

def round_down_to_nearest(number, bin_size=300):
    # Round down to the nearest multiple of 100
    rounded_number = (number // 100) * 100
    # Find the nearest multiple of 300 that is less than or equal to the rounded number
    rounded_number = (rounded_number // bin_size) * bin_size
    return rounded_number

def filter_calories(df, column='calories', quartile_percent=0.9):
    #get the calories cutoff and round up or down, whichever is closer
    calorie_cutoff = df[column].quantile(quartile_percent)
    rounded_down = round_down_to_nearest(calorie_cutoff)
    rounded_up = round_up_to_nearest(calorie_cutoff)

    diff_to_rounded_down = abs(calorie_cutoff - rounded_down)
    diff_to_rounded_up = abs(calorie_cutoff - rounded_up)
    
    # Select the min or max value, whichever is closer to the number
    if diff_to_rounded_down < diff_to_rounded_up:
        return df[df[column] < rounded_down]['calories']
    else:
        return df[df[column] < rounded_up]['calories']

def sorted_binned_encoding(series):
    """
    Input: Series of interval values in the form of strings, e.g. '0-299', '300-599', etc.
    Output: sorted dictionary with intervals as keys and integers as values, from 0 to the number of intervals. 
    """
    intervals = list(series.unique())
    sorted_intervals = sorted((map(lambda x: tuple(map(int, x.split('-'))), intervals)), key=lambda x: x[0])
    # Convert sorted tuples back to interval strings
    sorted_intervals = ['{}-{}'.format(lower, upper) for lower, upper in sorted_intervals]
    
    label_encoding_map = {}
    for i in range(len(sorted_intervals)):
        label_encoding_map[sorted_intervals[i]] = i
    
    return label_encoding_map

def collapsing_to_priority(type_lst, priority_list):
    """
    Collapse a list of items into just one based on a priority list.

    Parameters:
    - type_lst (list): A list of types to be collapsed into a single type.
    - priority_list (list): A list specifying the priority order of types.

    Returns:
    - str: The collapsed type based on the priority list.

    If the input list contains only one type, it is returned as is.
    Otherwise, the function iterates through the priority list
    and returns the first type found in the input list.
    If none of the types in the input list are found in the priority list,
    the function returns the first type from the priority list
    """
    if len(type_lst) == 1:
        return type_lst[0]
    else:
        for priority_item in priority_list:
            if priority_item in type_lst:
                return priority_item
        else:
            warnings.warn("{} was not found in the priority list, returning the first priority list item.".format(priority_item))
            return priority_list[0]

def priority_list_dish_type():
    priority_list = [
        'main course', 'starter', 'salad', 'soup', 'drinks', 'bread', 'desserts', 'condiments and sauces', 
        'sandwiches', 'cereals', 'alcohol cocktail', 'biscuits and cookies', 'pancake', 'egg', 'preserve',
        'omelet', 'special occasions', 'christmas', 'preps', 'thanksgiving', 'cinco de mayo'
    ]
    return priority_list

def priority_list_meal_type():
    priority_list = ['breakfast', 'lunch/dinner', 'brunch', 'snack', 'teatime']
    return priority_list

def one_hot_encode(df, column):
    onehot_encoder = OneHotEncoder()
    
    # Fit and transform the column to one-hot encoded format
    onehot_encoded = onehot_encoder.fit_transform(df[[column]])
    onehot_encoded_array = onehot_encoded.toarray()
    onehot_encoded_df = pd.DataFrame(onehot_encoded_array, columns=onehot_encoder.get_feature_names_out([column]))

    return onehot_encoded_df

#making a function to remove stop words - taken from the above link
def remove_stop_words(text, english_stop_words):
    #get the words in the review as a list
    text_words = text.split()
    
    #make a new list with the same words but only if they are not a stop word
    removed_stop_words_list = [word for word in text_words if word not in english_stop_words]
    removed_stop_words = ' '.join(removed_stop_words_list)
    
    return removed_stop_words

def lemmatization(text, lemmatizer):
    text_list = text.split()
    #lemmatize the words
    lemmatized_list = [lemmatizer.lemmatize(word) for word in text_list]
    #make it into a string again
    lemmatized_text = ' '.join(lemmatized_list)
    
    return lemmatized_text

def pre_process_text(df, column, stop_words, lemmatizer, tokenizer, inplace=False):
    recipes = df[column].copy()  # Make a copy to avoid modifying the original DataFrame inplace

    recipes = recipes.apply(lambda x: remove_stop_words(x, stop_words))
    recipes = recipes.apply(lambda x: lemmatization(x, lemmatizer))
    recipes = recipes.apply(lambda x: tokenizer(x))

    if inplace:
        df.loc[:, column] = recipes
        return None  # Return None if inplace=True to indicate that the DataFrame is modified inplace
    else:
        new_df = df.copy()  # Make a copy of the original DataFrame
        new_df.loc[:, column] = recipes
        return new_df