import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Set random seed for reproducibility
np.random.seed(42)

# ----------------------------------------------------------------------
# 1. Dataset Setup (Creating synthetic data with 3 distinct clusters)
# ----------------------------------------------------------------------
# X: feature data, y_true: true labels (we pretend we don't know this in unsupervised learning)
X, y_true = make_blobs(
    n_samples=300, 
    centers=3, 
    cluster_std=0.60, 
    random_state=0
)
print(f"Dataset created with {X.shape[0]} samples and {X.shape[1]} features.")
print("-" * 50)


# ----------------------------------------------------------------------
# 2. Method: Elbow Method (Choosing optimal K)
# ----------------------------------------------------------------------
print("--- 2. Elbow Method: Finding Optimal K ---")

wcss = [] # Within-Cluster Sum of Squares (Inertia)
k_range = range(1, 11)

# Run K-Means for K=1 to K=10 and record the WCSS (inertia)
for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

print("WCSS values calculated for K=1 to 10. (Look for the 'elbow' point in the plot)")

# In a real environment, you would plot this:
# plt.plot(k_range, wcss)
# plt.title('Elbow Method')
# plt.xlabel('Number of clusters (K)')
# plt.ylabel('WCSS')
# plt.show()
print("-" * 50)


# ----------------------------------------------------------------------
# 3. Method: Internal Metrics (Evaluating Cluster Quality)
# ----------------------------------------------------------------------
# We choose K=3 (which is visually the correct number and the elbow point)
k_optimal = 3
kmeans_optimal = KMeans(n_clusters=k_optimal, init='k-means++', n_init=10, random_state=42)
cluster_labels = kmeans_optimal.fit_predict(X)

print(f"--- 3. Evaluating Clustering with K={k_optimal} ---")

# 3a. Silhouette Score
# Score closer to 1.0 indicates better defined and separated clusters.
silhouette_avg = silhouette_score(X, cluster_labels)
print(f"Silhouette Score: {silhouette_avg:.4f}")

# 3b. Davies-Bouldin Index
# Score closer to 0.0 indicates better clustering (lower is better).
db_index = davies_bouldin_score(X, cluster_labels)
print(f"Davies-Bouldin Index: {db_index:.4f}")

# 3c. Calinski-Harabasz Index
# Score closer to infinity indicates better clustering (higher is better).
from sklearn.metrics import calinski_harabasz_score
ch_index = calinski_harabasz_score(X, cluster_labels)
print(f"Calinski-Harabasz Index: {ch_index:.4f}")
print("-" * 50)

# ----------------------------------------------------------------------
# 4. Example of Poor Clustering (K=2, visually incorrect)
# ----------------------------------------------------------------------
print("--- 4. Evaluating Poor Clustering with K=2 ---")
kmeans_poor = KMeans(n_clusters=2, init='k-means++', n_init=10, random_state=42)
labels_poor = kmeans_poor.fit_predict(X)

silhouette_poor = silhouette_score(X, labels_poor)
db_poor = davies_bouldin_score(X, labels_poor)
ch_poor = calinski_harabasz_score(X, labels_poor)

print(f"K=2 Silhouette Score: {silhouette_poor:.4f} (Lower than K=3: {silhouette_avg:.4f})")
print(f"K=2 Davies-Bouldin Index: {db_poor:.4f} (Higher than K=3: {db_index:.4f})")
print(f"K=2 Calinski-Harabasz Index: {ch_poor:.4f} (Lower than K=3: {ch_index:.4f})")
print("\nInterpretation: The internal metrics correctly indicate that K=3 provides a better cluster structure than K=2.")
