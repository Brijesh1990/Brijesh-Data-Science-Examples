import pandas as pd

chunk_size = 100000  # Load 100,000 rows at a time
chunks = pd.read_csv('massive_data.csv', chunksize=chunk_size)

# Process data in chunks (e.g., counting)
total_rows = 0
for chunk in chunks:
    total_rows += len(chunk)

print(f"Total rows processed: {total_rows}")
