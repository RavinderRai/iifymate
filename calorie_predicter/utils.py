"""
A collection of utility functions for data preprocessing, model training, and predicting.
"""
import warnings
import os
import yaml
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def round_up_to_nearest(number, bin_size=300):
    """
    Round up the input number to the nearest multiple of the specified bin size.
    Parameters:
        number (int or float): The number to be rounded up.
        bin_size (int, optional): The size of the bin to round up to. Defaults to 300.
    Returns:
        int: The rounded-up number.
    """
    # Round up to the nearest multiple of 100
    rounded_number = ((number + 99) // 100) * 100
    # Find the nearest multiple of 300 that is greater than or equal to the rounded number
    rounded_number = ((rounded_number + (bin_size-1)) // bin_size) * bin_size
    return rounded_number


def round_down_to_nearest(number, bin_size=300):
    """
    Round down the input number to the nearest multiple of the specified bin size.
    Parameters:
        number (int or float): The number to be rounded down.
        bin_size (int, optional): The size of the bin to round down to. Defaults to 300.
    Returns:
        int: The rounded-down number.
    """
    # Round down to the nearest multiple of 100
    rounded_number = (number // 100) * 100
    # Find the nearest multiple of 300 that is less than or equal to the rounded number
    rounded_number = (rounded_number // bin_size) * bin_size
    return rounded_number


def filter_calories(df, column='calories', quartile_percent=0.9):
    """
    Filter DataFrame based on the specified quartile percentage of calorie values.
    Parameters:
        df (DataFrame): The input DataFrame containing recipe data from the Edamam API.
        column (str, optional): The column containing calorie values. Defaults to 'calories'.
        quartile_percent (float, optional): The quartile percentage to filter the calorie values. 
                                            Defaults to 0.9.
    Returns:
        Series: Filtered calorie values based on the quartile percentage.
    """
    calorie_cutoff = df[column].quantile(quartile_percent)
    rounded_down = round_down_to_nearest(calorie_cutoff)
    rounded_up = round_up_to_nearest(calorie_cutoff)

    diff_to_rounded_down = abs(calorie_cutoff - rounded_down)
    diff_to_rounded_up = abs(calorie_cutoff - rounded_up)

    # Select the min or max value, whichever is closer to the number
    if diff_to_rounded_down < diff_to_rounded_up:
        return df[df[column] < rounded_down]['calories']
    return df[df[column] < rounded_up]['calories']


def sorted_binned_encoding(series):
    """
    Input: Series of interval values in the form of strings, e.g. '0-299', '300-599', etc.
    Output: sorted dictionary with intervals as keys and integers as values, 
            from 0 to the number of intervals. 
    """
    intervals = list(series.unique())
    sorted_intervals = sorted(
        (map(lambda x: tuple(map(int, x.split('-'))), intervals)), key=lambda x: x[0])
    # Convert sorted tuples back to interval strings
    sorted_intervals = ['{}-{}'.format(lower, upper)
                        for lower, upper in sorted_intervals]

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
            warnings.warn(
                "{} was not found in the priority list, returning the first priority list item.".format(priority_item))
            return priority_list[0]


def priority_list_dish_type():
    """
    The priority list for dish types, meant to collapse multilabel values to a single item.
    """
    priority_list = [
        'main course', 'starter', 'salad', 'soup', 'drinks', 'bread', 'desserts',
        'condiments and sauces', 'sandwiches', 'cereals', 'alcohol cocktail', 
        'biscuits and cookies', 'pancake', 'egg', 'preserve', 'omelet', 
        'special occasions', 'christmas', 'preps', 'thanksgiving', 'cinco de mayo'
    ]
    return priority_list


def priority_list_meal_type():
    """
    The priority list for meal types, meant to collapse multilabel values to a single item.
    """
    priority_list = ['breakfast', 'lunch/dinner', 'brunch', 'snack', 'teatime']
    return priority_list

def one_hot_encode(df, column):
    """
    Perform one-hot encoding on a specified column of a DataFrame.
    Parameters:
        df (DataFrame): The input DataFrame containing the column to be one-hot encoded.
        column (str): The name of the column to be one-hot encoded.
    Returns:
        tuple: A tuple containing the one-hot encoded DataFrame and the fitted OneHotEncoder object.
    """
    onehot_encoder = OneHotEncoder()
    onehot_encoder.fit(df[[column]])

    # Transform the column to one-hot encoded format
    onehot_encoded = onehot_encoder.transform(df[[column]])
    onehot_encoded_array = onehot_encoded.toarray()
    onehot_encoded_df = pd.DataFrame(
        onehot_encoded_array,
        columns=onehot_encoder.get_feature_names_out([column])
    )

    return onehot_encoded_df, onehot_encoder

def remove_stop_words(text, english_stop_words):
    """
    Remove stop words from a given text.
    Parameters:
        text (str): The input text to be processed.
        english_stop_words (list): A list of English stop words.
    Returns:
        str: The processed text with stop words removed.
    """
    # get the words in the review as a list
    text_words = text.split()

    # make a new list with the same words but only if they are not a stop word
    removed_stop_words_list = [
        word for word in text_words if word not in english_stop_words]
    removed_stop_words = ' '.join(removed_stop_words_list)

    return removed_stop_words


def lemmatization(text, lemmatizer):
    """
    Lemmatize the words in a given text.
    Parameters:
        text (str): The input text to be lemmatized.
        lemmatizer: A lemmatizer object.
    Returns:
        str: The lemmatized text.
    """
    text_list = text.split()
    # lemmatize the words
    lemmatized_list = [lemmatizer.lemmatize(word) for word in text_list]
    # make it into a string again
    lemmatized_text = ' '.join(lemmatized_list)

    return lemmatized_text


def pre_process_text(df, column, stop_words, lemmatizer, tokenizer, inplace=False):
    """
    Pre-process text data in a DataFrame column.
    Parameters:
        df (DataFrame): The input DataFrame containing the text column to be pre-processed.
        column (str): The name of the text column to be pre-processed.
        stop_words (list): A list of stop words to be removed from the text.
        lemmatizer: A lemmatizer object.
        tokenizer: A tokenizer function.
        inplace (bool, optional): If True, modify the original DataFrame inplace. Default is False.
    Returns:
        DataFrame or None: The pre-processed DataFrame if inplace=False, None otherwise.
    """
    recipes = df[column].copy()
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


def get_experiment_folder_path(target_experiment_name, mlflow_dir="mlruns"):
    """
    Get the path to the folder corresponding to a specific MLflow experiment.

    Parameters:
        target_experiment_name (str): The name of the target MLflow experiment.
        mlflow_dir (str, optional): The directory containing MLflow experiments.
                                    Defaults to "mlruns".

    Returns:
        str: The path to the folder of the specified MLflow experiment,
             or None if the experiment is not found.

    """
    # Iterate through each folder in mlruns, then check the YAML file for the experiment ID
    for folder_name in os.listdir(mlflow_dir):
        folder_path = os.path.join(mlflow_dir, folder_name)
        if os.path.isdir(folder_path):
            yaml_file_path = os.path.join(folder_path, "meta.yaml")
            if os.path.exists(yaml_file_path):
                # Open the YAML file and read the experiment name
                with open(yaml_file_path, "r") as yaml_file:
                    meta_data = yaml.safe_load(yaml_file)
                    experiment_name = meta_data.get("name")
                    if experiment_name == target_experiment_name:
                        return folder_path
    # Experiment not found, issue a warning
    warnings.warn(f"Experiment '{target_experiment_name}' not found.")
    return None


def sample_data(df, target='calories', sample_size=3, to_csv=False):
    """
    Samples a subset of the DataFrame based on floored values of the target column.

    Parameters:
        df (DataFrame): Input DataFrame containing recipe data.
        target (str): Name of the target column to floor values on. Default is 'calories'.
        sample_size (int): Number of samples to select for each floored value. Default is 3.
        to_csv (bool): If True, saves the sampled DataFrame to 'sample_recipes.csv'. Else returns sample_df. 
    Returns:
        DataFrame or None: Sampled subset of the DataFrame if `to_csv` is False, None otherwise.
    """
    floored_calories = df[target].apply(
        lambda x: x // 100 * 100).rename('floored_calories')
    df_floored_calories = pd.concat([df, floored_calories], axis=1)
    sample_df = df_floored_calories.groupby(
        'floored_calories', group_keys=False).head(sample_size)
    sample_df = sample_df.drop('floored_calories', axis=1)

    if to_csv:
        sample_df.to_csv('sample_recipes.csv', index=False)
        return None
    else:
        return sample_df
