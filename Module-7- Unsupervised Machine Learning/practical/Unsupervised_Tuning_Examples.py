import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Set random seed for reproducibility
np.random.seed(42)

# ----------------------------------------------------------------------
# 1. Dataset Setup (Creating synthetic data with 4 distinct clusters)
# ----------------------------------------------------------------------
X, y_true = make_blobs(
    n_samples=500, 
    centers=4, 
    cluster_std=0.8, 
    random_state=42
)
print(f"Dataset created with {X.shape[0]} samples. (True clusters: 4)")
print("-" * 50)


# ----------------------------------------------------------------------
# 2. Hyperparameter Tuning K-Means: The Elbow Method (Optimizing WCSS)
# ----------------------------------------------------------------------
print("--- 2. Tuning K-Means: Elbow Method (Optimizing Inertia) ---")

wcss = [] # WCSS (Inertia) tracks the sum of squared distances to the nearest cluster center
k_range = range(1, 11)

for k in k_range:
    # Set n_init to 'auto' or explicit number to suppress warnings
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# In a real scenario, you plot wcss vs k_range to find the bend/elbow.
print(f"WCSS values: {np.round(wcss, 2)}")
print("Optimal K is where the drop in WCSS slows (visual elbow). Here, it should be K=4.")
print("-" * 50)


# ----------------------------------------------------------------------
# 3. Hyperparameter Tuning K-Means: Silhouette Method (Optimizing Score)
# ----------------------------------------------------------------------
print("--- 3. Tuning K-Means: Silhouette Method (Optimizing Quality) ---")

silhouette_scores = {}
# Cannot calculate Silhouette for K=1, so range starts at 2
k_range_silhouette = range(2, 11) 

for k in k_range_silhouette:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=42)
    labels = kmeans.fit_predict(X)
    
    # Calculate the mean Silhouette Score for the current clustering result
    score = silhouette_score(X, labels)
    silhouette_scores[k] = score

# Find the K that maximized the score
best_k = max(silhouette_scores, key=silhouette_scores.get)

print(f"Silhouette Scores: {silhouette_scores}")
print(f"The best K (highest score) is K={best_k} (Score: {silhouette_scores[best_k]:.4f})")
print("This method confirms the visual choice from the Elbow method (K=4).")
