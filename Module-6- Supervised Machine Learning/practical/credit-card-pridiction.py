import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Assuming you have loaded your data into X (features) and y (target)

# --- 1. Credit Risk Example ---
# X would include: Income, Credit Score, Debt-to-Income, etc.
# y would include: 'Default' or 'Repay'

# --- 2. Customer Segmentation Example ---
# X would include: Age, AOV, Visits/Month, etc.
# y would include: 'Premium', 'Bargain Hunter', 'Standard Loyal'

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create the Decision Tree Classifier model
model = DecisionTreeClassifier(max_depth=5) # max_depth helps prevent overfitting

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
