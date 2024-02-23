import pandas as pd
import ast
from processing_functions import round_up_to_nearest, round_down_to_nearest, filter_calories, sorted_binned_encoding

raw_df = pd.read_csv('../recipes.csv')

#dropping duplicates from recipe name (which is the label column) because some sources give the same recipes
df = raw_df.drop_duplicates('label')

#handling target variable first
calories_df = df['calories']

#capping the calorie count, so we will include recipes with calorie counts so that we maintain 90% of our data
filtered_calories_df = filter_calories(df, column='calories', quartile_percent=0.9)
max_calorie_cutoff = round_up_to_nearest(max(filtered_calories_df))

#binning the calorie count to turn this into a classification problem
bin_edges = [i for i in range(0, int(max_calorie_cutoff)+1, 300)]
labels = [f"{bin_edges[i]}-{bin_edges[i+1]-1}" for i in range(len(bin_edges)-1)]
binned_calories = pd.cut(filtered_calories_df, bins=bin_edges, labels=labels, include_lowest=True)

# Assuming binned_calories is a pandas Series or DataFrame
assert binned_calories.isna().sum() == 0, "The count of NaN values in binned_calories is not equal to 0"

binned_calories = pd.DataFrame(binned_calories).rename(columns={'calories':'binnedCalories'})
label_encoding = sorted_binned_encoding(binned_calories)
target = binned_calories.map(label_encoding)

binned_calories_df = df.loc[target.index]

print(binned_calories_df.shape)
