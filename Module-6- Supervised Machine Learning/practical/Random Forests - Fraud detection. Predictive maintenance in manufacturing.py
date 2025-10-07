from sklearn.ensemble import RandomForestClassifier # Use Regressor for maintenance time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Load data (X = Features, y = Target)
# ...

# Using dummy data for demonstration
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=100, n_features=20, random_state=42)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create the Random Forest model (n_estimators is the number of trees)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_features='sqrt')

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions (e.g., for Fraud Detection)
y_pred = rf_model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))

# Access Feature Importance (A major advantage of Random Forests)
feature_importances = pd.Series(rf_model.feature_importances_, index=pd.DataFrame(X).columns).sort_values(ascending=False)
print("\nTop Feature Importances:\n", feature_importances.head())