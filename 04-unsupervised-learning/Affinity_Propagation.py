"""
Affinity Propagation Clustering
- No need to specify the number of clusters in advance (unlike K-Means)
- Data points exchange "messages" with each other, and exemplars (cluster centers) get selected
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

# 1. Generate and scale the data
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)
X = StandardScaler().fit_transform(X)

# 2. Build the model
# damping: between 0.5 and 1 (higher value = more stable, but slower convergence)
# preference: lower value = fewer clusters, higher value = more clusters
model = AffinityPropagation(damping=0.9, preference=-50, random_state=42)
labels = model.fit_predict(X)

centers = model.cluster_centers_
n_clusters = len(centers)

# 3. Evaluate the clustering
print(f"Clusters found     : {n_clusters}")
print(f"Iterations         : {model.n_iter_}")
print(f"Silhouette score   : {silhouette_score(X, labels):.3f}")
print(f"Adjusted Rand index: {adjusted_rand_score(y_true, labels):.3f}")

# 4. Plot the clusters and exemplars
plt.figure(figsize=(7, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=30)
plt.scatter(centers[:, 0], centers[:, 1], c="red", marker="X", s=200,
            edgecolor="black", label="Exemplars")
plt.title(f"Affinity Propagation ({n_clusters} clusters)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.tight_layout()
plt.show()

# 5. Effect of the preference parameter on the number of clusters
for p in [-200, -100, -50, -10]:
    m = AffinityPropagation(damping=0.9, preference=p, random_state=42).fit(X)
    print(f"preference={p:>5} -> clusters={len(m.cluster_centers_)}")
