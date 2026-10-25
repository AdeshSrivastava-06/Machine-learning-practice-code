"""
BIRCH Clustering
(Balanced Iterative Reducing and Clustering using Hierarchies)

BIRCH builds a compact tree summary (Clustering Feature Tree) of the data in a
single pass, which makes it fast and memory-efficient for large datasets.

Key parameters:
    threshold         - max radius of a sub-cluster; smaller = more sub-clusters
    branching_factor  - max number of sub-clusters per tree node
    n_clusters        - final number of clusters (None = keep sub-clusters)
"""

import os
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import Birch
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# 1. Data


X, y_true = make_blobs(
    n_samples=1000, centers=4, cluster_std=0.8, random_state=42
)
X = StandardScaler().fit_transform(X)


# 2. Fit BIRCH

birch = Birch(threshold=0.5, branching_factor=50, n_clusters=4)
labels = birch.fit_predict(X)

print(f"Clusters found      : {len(np.unique(labels))}")
print(f"Sub-clusters in tree: {len(birch.subcluster_centers_)}")
print(f"Silhouette score    : {silhouette_score(X, labels):.3f}")


# 3. Effect of the threshold parameter

print("\nThreshold sweep:")
for t in [0.1, 0.3, 0.5, 1.0, 2.0]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # hide "fewer sub-clusters than n_clusters" warning
        model = Birch(threshold=t, n_clusters=4).fit(X)

    n_sub = len(model.subcluster_centers_)
    n_found = len(np.unique(model.labels_))

    # silhouette_score needs at least 2 clusters; a large threshold can
    # merge everything into 1 sub-cluster, so guard against that.
    if n_found < 2:
        print(f"  threshold={t:<4} sub-clusters={n_sub:<4} "
              f"silhouette=N/A (only {n_found} cluster - threshold too large)")
    else:
        score = silhouette_score(X, model.labels_)
        print(f"  threshold={t:<4} sub-clusters={n_sub:<4} "
              f"clusters={n_found} silhouette={score:.3f}")


# 4. Visualise

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=15)
axes[0].set_title("BIRCH final clusters")

sub = birch.subcluster_centers_
axes[1].scatter(X[:, 0], X[:, 1], c="lightgray", s=10)
axes[1].scatter(sub[:, 0], sub[:, 1], c="red", s=25, marker="x")
axes[1].set_title("BIRCH sub-cluster centers (CF tree leaves)")

for ax in axes:
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

plt.tight_layout()
plt.show()


# 5. Incremental learning (BIRCH's special ability)
# partial_fit lets you stream data in chunks without reloading everything.

stream_model = Birch(threshold=0.5, n_clusters=4)
for chunk in np.array_split(X, 5):
    stream_model.partial_fit(chunk)

print(f"\nStreaming fit sub-clusters: {len(stream_model.subcluster_centers_)}")
