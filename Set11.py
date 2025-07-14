import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Create a dummy CSV file for demonstration purposes
# In a real scenario, you would upload your file or provide the correct path
data = {'Math': [70, 85, 60, None, 95, 75, 80, 65, 90, 55],
        'Science': [75, 80, 65, 70, 90, 85, None, 60, 92, 58],
        'English': [80, 75, None, 88, 92, 70, 85, 62, 95, 50]}
df_dummy = pd.DataFrame(data)
file_path = 'SET-11.csv'
df_dummy.to_csv(file_path, index=False)

# 1. Load the CSV file
df = pd.read_csv(file_path)

# 2. Replace missing values in Math and Science with the mean
df['Math'] = df['Math'].fillna(df['Math'].mean())
df['Science'] = df['Science'].fillna(df['Science'].mean())

# 3. Remove duplicate entries
df = df.drop_duplicates()

# 4. Normalize numerical values using Min-Max Normalization
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# 5. Discretize Science marks into categories
# Load original values to reverse normalization for correct binning
# original_df = pd.read_csv(file_path) # No need to reload, use df before normalization
original_science = df_dummy['Science'].fillna(df_dummy['Science'].mean()) # Use df_dummy to get original values
min_science = original_science.min()
max_science = original_science.max()

# Define the categorization function
def categorize_science(score):
    if score < 50:
        return 'Poor'
    elif score < 70:
        return 'Average'
    elif score < 90:
        return 'Good'
    else:
        return 'Excellent'

# Reverse normalization for Science and apply category
# We need the original values for correct categorization, not the normalized ones
# Let's create a new column with original science values for categorization before normalization
df['Science_Original'] = df_dummy['Science'].fillna(df_dummy['Science'].mean())
df['Science_Category'] = df['Science_Original'].apply(categorize_science)
df = df.drop(columns=['Science_Original']) # Drop the temporary column


# 6. Smooth noisy data using binning (equal-width binning) on Math
# Use the Math column after filling missing values but before normalization for meaningful binning
df_for_binning = pd.read_csv(file_path)
df_for_binning['Math'] = df_for_binning['Math'].fillna(df_for_binning['Math'].mean())

df_sorted = df.sort_values(by='Math').copy()
bins = pd.cut(df_sorted['Math'], bins=4, labels=False)
df_sorted['Math_Binned'] = bins
df_sorted['Math_Smoothed'] = df_sorted.groupby('Math_Binned')['Math'].transform('mean')
df_sorted.drop(columns=['Math_Binned'], inplace=True)

# Merge the smoothed Math column back to the original dataframe
df = df.merge(df_sorted[['Math_Smoothed']], left_index=True, right_index=True)


# Display final DataFrame
print(df.head())
