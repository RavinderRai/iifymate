import pandas as pd

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

def sorted_binned_encoding(df):
    """
    Input: dataframe with 1 column or series of interval values in the form of strings, e.g. '0-299', '300-599', etc.
    Output: sorted dictionary with intervals as keys and integers as values, from 0 to the number of intervals. 
    """
    intervals = list(df.nunique())
    sorted_intervals = sorted((map(lambda x: tuple(map(int, x.split('-'))), intervals)), key=lambda x: x[0])
    # Convert sorted tuples back to interval strings
    sorted_intervals = ['{}-{}'.format(lower, upper) for lower, upper in sorted_intervals]
    
    label_encoding_map = {}
    for i in range(len(sorted_intervals)):
        label_encoding_map[sorted_intervals[i]] = i
    
    return label_encoding_map