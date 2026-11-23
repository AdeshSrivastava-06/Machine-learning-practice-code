# Cluster stability analysis using repeated subsampling.
# Goal: check how stable the clusters are for each value of k, without using
# the true labels. A good clustering should not change much when the data
# changes a little.
# Method: for every k, take two different random 80 percent subsamples of the
# data, fit KMeans on each one, and use both models to label ALL points.
# The two labelings are compared with the Adjusted Rand Index (ARI).
# ARI of 1.0 means both runs found the same clusters, and a value near 0
# means the clusters changed completely. This is repeated 30 times per k.
# Data: Wine dataset (13 features, 3 true classes).

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

X, y = load_wine(return_X_y=True)
X = StandardScaler().fit_transform(X)
n = len(X)
print("Dataset:", n, "rows,", X.shape[1], "features, true classes: 3")
print()

REPEATS = 30
FRACTION = 0.8

print(f"{'k':>2s} {'Mean ARI':>9s} {'Std':>7s} {'Lowest':>8s}")
results = {}
for k in range(2, 9):
    scores = []
    for run in range(REPEATS):
        rng = np.random.RandomState(run)
        idx_a = rng.choice(n, int(FRACTION * n), replace=False)
        idx_b = rng.choice(n, int(FRACTION * n), replace=False)
        model_a = KMeans(n_clusters=k, n_init=5, random_state=run).fit(X[idx_a])
        model_b = KMeans(n_clusters=k, n_init=5, random_state=run + 1000).fit(X[idx_b])
        # Both models label all points, so the two labelings can be compared
        scores.append(adjusted_rand_score(model_a.predict(X), model_b.predict(X)))
    results[k] = scores
    print(f"{k:2d} {np.mean(scores):9.4f} {np.std(scores):7.4f} {np.min(scores):8.4f}")

# Summary
means = {k: np.mean(v) for k, v in results.items()}
ranked = sorted(means, key=means.get, reverse=True)
print()
print("Summary:")
print(f"Most stable k: {ranked[0]} (mean ARI {means[ranked[0]]:.3f}), "
      f"then k = {ranked[1]} ({means[ranked[1]]:.3f}).")
print("The true number of classes is 3.")
if ranked[0] != 3:
    print(f"Stability alone chose k = {ranked[0]}, not 3. Stability tends to favour small k,")
    print("because splitting into only two big groups is easy to repeat.")
else:
    print("Stability picked the true number of classes.")
print("Stability measures repeatability, not correctness, so it should be used")
print("together with other checks and with knowledge about the data.")
