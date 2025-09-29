import pandas as pd # 1. Import the pandas library, commonly aliased as 'pd'

# 2. Define the file path
file_path = 'sales_data.csv'

# 3. Read the CSV file into a DataFrame
# 'df' is the conventional variable name for a pandas DataFrame
try:
    df = pd.read_csv(
        file_path,
        sep=',',               # Specify the delimiter (default is comma)
        header=0,              # Specifies the row number to use as column names (0-indexed)
        encoding='utf-8',      # Character encoding (often 'utf-8' or 'latin-1')
        na_values=['N/A', '?'] # Values in the file to be treated as missing (NaN)
    )
    
    # 4. Display the first few rows to confirm successful import
    print("Data imported successfully!")
    print(df.head())
    
except FileNotFoundError:
    print(f"Error: The file {file_path} was not found.")
except Exception as e:
    print(f"An error occurred during import: {e}")