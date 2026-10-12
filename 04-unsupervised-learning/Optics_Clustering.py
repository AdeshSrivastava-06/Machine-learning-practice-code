"""
OPTICS (Ordering Points To Identify the Clustering Structure)

A density-based clustering algorithm similar to DBSCAN, but it can find
clusters of varying density and does not need a fixed `eps`.
It produces a reachability plot from which clusters are extracted.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


# 1. Create data with clusters of different densities

X, _ = make_blobs(
    n_samples=[300, 300, 300],
    centers=[(-5, -5), (0, 0), (6, 6)],
    cluster_std=[0.4, 1.0, 1.5],
    random_state=42,
)
X = StandardScaler().fit_transform(X)


# 2. Fit OPTICS

optics = OPTICS(min_samples=20, xi=0.05, min_cluster_size=0.05)
labels = optics.fit_predict(X)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print(f"Clusters found : {n_clusters}")
print(f"Noise points   : {n_noise}")


# 3. Visualize: reachability plot + clustering result

space = np.arange(len(X))
reachability = optics.reachability_[optics.ordering_]
ordered_labels = labels[optics.ordering_]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Reachability plot
colors = plt.cm.tab10(np.linspace(0, 1, max(n_clusters, 1)))
for k, color in zip(range(n_clusters), colors):
    mask = ordered_labels == k
    ax1.plot(space[mask], reachability[mask], ".", color=color, alpha=0.6)
ax1.plot(space[ordered_labels == -1], reachability[ordered_labels == -1],
         "k.", alpha=0.3)
ax1.set_title("Reachability Plot")
ax1.set_xlabel("Ordered samples")
ax1.set_ylabel("Reachability distance")

# Cluster scatter
for k, color in zip(range(n_clusters), colors):
    ax2.scatter(*X[labels == k].T, color=color, s=15, label=f"Cluster {k}")
ax2.scatter(*X[labels == -1].T, c="k", s=10, alpha=0.4, label="Noise")
ax2.set_title("OPTICS Clustering")
ax2.legend()

plt.tight_layout()
plt.show()
