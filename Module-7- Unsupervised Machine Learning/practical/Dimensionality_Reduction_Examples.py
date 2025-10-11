import numpy as np
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# ----------------------------------------------------------------------
# 1. Dataset Setup (Creating synthetic high-dimensional data)
# ----------------------------------------------------------------------
N_SAMPLES = 500
N_FEATURES = 50 # High dimension
N_COMPONENTS_TRUE = 5 # Assume true inherent dimension is low

# Create synthetic data where only 5 features truly matter
X, y = make_blobs(
    n_samples=N_SAMPLES, 
    n_features=N_FEATURES, 
    centers=4, 
    cluster_std=1.0, 
    random_state=42
)
print(f"Original Dataset Shape: {X.shape}")
print("-" * 50)


# ----------------------------------------------------------------------
# 2. Technique: Principal Component Analysis (PCA)
# ----------------------------------------------------------------------
print("--- 2. PCA for Feature Transformation and Compression ---")

# A. Determine Optimal Number of Components (Explained Variance)
# We fit PCA to see how much variance each component explains
pca_full = PCA()
pca_full.fit(X)

# Calculate cumulative explained variance
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

# Find number of components to explain 90% of variance
n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
print(f"Components needed to retain 90% variance: {n_components_90}")
print(f"Variance retained by 2 components (for visualization): {cumulative_variance[1]:.4f}")

# B. Transform the data to 2 dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print(f"PCA Reduced Dataset Shape: {X_pca.shape}")
print("-" * 50)


# ----------------------------------------------------------------------
# 3. Technique: t-SNE (for Visualization)
# ----------------------------------------------------------------------
print("--- 3. t-SNE for Non-linear Visualization ---")

# t-SNE is generally much slower than PCA, so we use a small subset 
# of the data or use PCA output as input for speed.
# Here we use the original high-dimensional data (X) for a pure demonstration.
# Note: n_jobs=-1 utilizes all available cores for faster computation.
tsne = TSNE(
    n_components=2, 
    perplexity=30,  # Hyperparameter controlling balance between local/global structure
    n_iter=300, 
    learning_rate='auto', 
    init='pca', 
    random_state=42
)
X_tsne = tsne.fit_transform(X)

print(f"t-SNE Reduced Dataset Shape: {X_tsne.shape}")
print("-" * 50)

# In a real environment, you would plot these results:
# fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
# axes[0].set_title('Data Reduced by PCA (Linear)')
# axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
# axes[1].set_title('Data Reduced by t-SNE (Non-Linear)')
# plt.show()
print("Interpretation: PCA provides feature vectors preserving variance, ideal for compression.")
print("t-SNE provides a scattered map preserving local clusters, ideal for visualization.")
