import pandas as pd
import numpy as np

# Create sample data with missing values
data = {'A': [1, 2, np.nan, 4], 'B': [5, np.nan, 7, 8]}
df = pd.DataFrame(data)

# Drop rows where ANY value is missing
df_dropped_rows = df.dropna(axis=0, how='any')

# Output: Only the first and last rows remain
# print(df_dropped_rows)
#    A  B
# 0 1.0 5
# 3 4.0 8