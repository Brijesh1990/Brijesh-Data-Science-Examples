import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Sample Data
data = pd.DataFrame({
    'Date': pd.to_datetime(['2025-01-01', '2025-01-02', '2025-01-03']),
    'Sales': [100, 150, 120],
    'Category': ['A', 'B', 'A'],
    'Price': [10.5, 20.2, 12.0]
})

# Line Plot Example (Time Series)
plt.figure(figsize=(6, 4))
plt.plot(data['Date'], data['Sales'], marker='o')
plt.title('Daily Sales Trend')
plt.xlabel('Date')
plt.ylabel('Sales Revenue')
plt.show()
