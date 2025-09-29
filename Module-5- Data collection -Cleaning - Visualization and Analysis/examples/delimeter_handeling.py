import pandas as pd

# 1. Reading a Semicolon-Separated File (often called a 'SSV')
df_ssv = pd.read_csv('euro_sales.csv', sep=';')
print("Loaded with Semicolon Delimiter:")
print(df_ssv.head(2))

# 2. Reading a Tab-Separated File (TSV)
df_tsv = pd.read_csv('database_export.tsv', sep='\t')
print("\nLoaded with Tab Delimiter:")
print(df_tsv.head(2))
