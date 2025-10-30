import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from scipy import stats

# ----------------------------------------------------------------------
# 1. Dataset Setup
# ----------------------------------------------------------------------
print("--- 1. Initial Dataset ---")
data = {
    'Salary': [50000, 60000, 75000, 150000, 55000],  # Numerical feature with an outlier
    'Age': [25, 30, 35, 40, 28],                     # Numerical feature
    'City': ['NY', 'LA', 'NY', 'SF', 'LA'],           # Categorical feature
    'Experience': [2, 5, 7, 10, 3]                    # Numerical feature
}
df = pd.DataFrame(data)
print(df)
print("-" * 30)

# ----------------------------------------------------------------------
# 2. Encoder (One-Hot Encoding)
# ----------------------------------------------------------------------
# We will encode the 'City' column.

print("--- 2. Encoder (One-Hot Encoding on 'City') ---")

# 1. Prepare the data: One-Hot Encoder expects the data to be 2D (a column).
# We reshape the 'City' column from (5,) to (5, 1).
city_column = df[['City']]

# 2. Initialize the encoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# 3. Fit and transform the data
encoded_cities = encoder.fit_transform(city_column)

# 4. Convert back to a DataFrame for readability
encoded_df = pd.DataFrame(
    encoded_cities,
    columns=encoder.get_feature_names_out(['City'])
)

# 5. Combine with the original dataframe (dropping the original 'City' column)
df_encoded = pd.concat([df.drop('City', axis=1), encoded_df], axis=1)
print(df_encoded)
print("Shape after encoding:", df_encoded.shape)
print("-" * 30)


# ----------------------------------------------------------------------
# 3. Scaler (StandardScaler)
# ----------------------------------------------------------------------
# We will scale the 'Salary' column to have a mean of 0 and a standard deviation of 1.

print("--- 3. Scaler (StandardScaler on 'Salary') ---")

# 1. Prepare the data (needs to be 2D)
salary_column = df[['Salary']]

# 2. Initialize the scaler
scaler = StandardScaler()

# 3. Fit and transform the data
scaled_salaries = scaler.fit_transform(salary_column)

# 4. Replace the original column with the scaled data
df_scaled = df.copy()
df_scaled['Salary_Scaled'] = scaled_salaries

print(df_scaled[['Salary', 'Salary_Scaled']])
print(f"Original Mean: {df_scaled['Salary'].mean():.2f}")
print(f"Scaled Mean (approx 0): {df_scaled['Salary_Scaled'].mean():.2f}")
print("-" * 30)

# ----------------------------------------------------------------------
# 4. Outlier Detection
# ----------------------------------------------------------------------
# We will detect the outlier in the 'Salary' column (150000 is far from the others).

print("--- 4. Outlier Detection on 'Salary' ---")

# Method A: Z-Score (Statistical Method)
print("\n-- 4a. Z-Score Method --")
# Calculate Z-scores for the 'Salary' column
z_scores = stats.zscore(df['Salary'])
df['Z_Score'] = z_scores

# Define a threshold (common threshold is 3)
z_score_threshold = 3
outliers_zscore = df[np.abs(df['Z_Score']) > z_score_threshold]

print(df[['Salary', 'Z_Score']])
print(f"\nOutliers based on Z > {z_score_threshold}:")
print(outliers_zscore)
# Note: In this small sample, the Z-score for 150000 might not exceed 3,
# but it illustrates the calculation.

# Method B: Isolation Forest (Model-Based Method)
print("\n-- 4b. Isolation Forest Method --")

# We use the numerical columns for this model
X = df[['Salary', 'Age', 'Experience']]

# Initialize Isolation Forest (contamination is the expected proportion of outliers)
iso_forest = IsolationForest(contamination=0.2, random_state=42)

# Fit and predict. '1' means inlier, '-1' means outlier.
outlier_predictions = iso_forest.fit_predict(X)

# Add the prediction to the DataFrame
df['Is_Outlier'] = np.where(outlier_predictions == -1, 'Yes', 'No')

print(df[['Salary', 'Is_Outlier']])
print("\nIdentified Outliers (Is_Outlier == 'Yes'):")
print(df[df['Is_Outlier'] == 'Yes'])
print("-" * 30)
