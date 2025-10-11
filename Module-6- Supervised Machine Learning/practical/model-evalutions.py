import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score, confusion_matrix
)
from sklearn.datasets import make_classification, make_regression

# Set random seed for reproducibility
np.random.seed(42)

# ----------------------------------------------------------------------
# A. CLASSIFICATION MODEL EVALUATION
# ----------------------------------------------------------------------
print("--- A. Classification Model Evaluation ---")

# 1. Create a synthetic classification dataset
X_cls, y_cls = make_classification(
    n_samples=500, 
    n_features=10, 
    n_classes=2, 
    random_state=42
)
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_cls, y_cls, test_size=0.3, random_state=42
)

# 2. Train a simple model (Logistic Regression)
model_cls = LogisticRegression()
model_cls.fit(X_train_cls, y_train_cls)

# 3. Make predictions
y_pred_cls = model_cls.predict(X_test_cls)
# Predict probabilities for ROC AUC
y_proba_cls = model_cls.predict_proba(X_test_cls)[:, 1]

# 4. Calculate Metrics
print("\n[Metrics]")
print(f"Accuracy: {accuracy_score(y_test_cls, y_pred_cls):.4f}")
print(f"Precision: {precision_score(y_test_cls, y_pred_cls):.4f}")
print(f"Recall: {recall_score(y_test_cls, y_pred_cls):.4f}")
print(f"F1-Score: {f1_score(y_test_cls, y_pred_cls):.4f}")
print(f"ROC AUC Score: {roc_auc_score(y_test_cls, y_proba_cls):.4f}")

# 5. Confusion Matrix
print("\n[Confusion Matrix (Actual vs Predicted)]")
cm = confusion_matrix(y_test_cls, y_pred_cls)
print(f"TN: {cm[0, 0]}, FP: {cm[0, 1]}")
print(f"FN: {cm[1, 0]}, TP: {cm[1, 1]}")
print("-" * 50)


# ----------------------------------------------------------------------
# B. REGRESSION MODEL EVALUATION
# ----------------------------------------------------------------------
print("--- B. Regression Model Evaluation ---")

# 1. Create a synthetic regression dataset
X_reg, y_reg = make_regression(
    n_samples=500, 
    n_features=5, 
    noise=10, 
    random_state=42
)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

# 2. Train a simple model (Linear Regression)
model_reg = LinearRegression()
model_reg.fit(X_train_reg, y_train_reg)

# 3. Make predictions
y_pred_reg = model_reg.predict(X_test_reg)

# 4. Calculate Metrics
print("\n[Metrics]")
mae = mean_absolute_error(y_test_reg, y_pred_reg)
mse = mean_squared_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_reg, y_pred_reg)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²): {r2:.4f}")
print("-" * 50)

# ----------------------------------------------------------------------
# C. Cross-Validation (Ensuring robust evaluation)
# ----------------------------------------------------------------------
print("--- C. Cross-Validation Example (Robust Evaluation) ---")
from sklearn.model_selection import cross_val_score

# Use the classification model and dataset
model = LogisticRegression(max_iter=1000)

# Perform 5-fold cross-validation using the 'accuracy' metric
scores = cross_val_score(model, X_cls, y_cls, cv=5, scoring='accuracy')

print(f"5-Fold Cross-Validation Scores: {scores}")
print(f"Average Accuracy across all folds: {scores.mean():.4f}")
print("Interpretation: Cross-validation gives a more reliable estimate of model performance.")
