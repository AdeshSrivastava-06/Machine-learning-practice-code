"""
HDBSCAN (Hierarchical Density-Based Spatial Clustering)

Improved version of DBSCAN. Instead of one fixed density threshold
(eps in DBSCAN), it builds a hierarchy of clusters across ALL density
levels and picks the most stable ones automatically. This means:
  - No need to tune "eps" (DBSCAN's most annoying parameter)
  - Handles clusters of VARYING density in the same dataset
  - Still naturally flags noise/outliers as -1

Requires scikit-learn >= 1.3 (HDBSCAN is built in, no extra install).
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import HDBSCAN, DBSCAN
from sklearn.datasets import make_blobs


def generate_data():
    # Clusters with deliberately DIFFERENT densities + noise
    X1, _ = make_blobs(n_samples=200, centers=[[0, 0]], cluster_std=0.3, random_state=42)
    X2, _ = make_blobs(n_samples=200, centers=[[5, 5]], cluster_std=1.2, random_state=42)
    rng = np.random.RandomState(42)
    noise = rng.uniform(low=-3, high=8, size=(30, 2))

    X = np.vstack([X1, X2, noise])
    return X


def run_hdbscan(X, min_cluster_size=15, min_samples=5):
    """
    min_cluster_size: smallest grouping that still counts as a cluster.
    min_samples: how conservative the algorithm is about calling a point
                 "noise" - higher value -> more points labeled as noise.
    """
    model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = model.fit_predict(X)
    return labels


def compare_with_dbscan(X):
    hdbscan_labels = run_hdbscan(X)
    dbscan_labels = DBSCAN(eps=0.5, min_samples=5).fit_predict(X)  # single eps struggles here

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(X[:, 0], X[:, 1], c=dbscan_labels, cmap="tab10", s=20)
    axes[0].set_title("DBSCAN (fixed eps - misses varying density)")

    axes[1].scatter(X[:, 0], X[:, 1], c=hdbscan_labels, cmap="tab10", s=20)
    axes[1].set_title("HDBSCAN (adapts to varying density)")

    plt.tight_layout()
    plt.savefig("hdbscan_vs_dbscan.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    X = generate_data()

    labels = run_hdbscan(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)

    print(f"Clusters found: {n_clusters}")
    print(f"Noise points: {n_noise}")

    compare_with_dbscan(X)
