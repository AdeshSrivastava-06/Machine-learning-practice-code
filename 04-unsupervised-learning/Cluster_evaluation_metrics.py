# Cluster evaluation metrics on the Wine dataset.
# Goal 1: choose the number of clusters for KMeans using internal metrics,
# which do not need the true labels.
# Goal 2: compare KMeans, Agglomerative clustering and Gaussian Mixture on
# the same data, using both internal metrics and the true labels.
# Silhouette and Calinski-Harabasz: higher is better.
# Davies-Bouldin and inertia: lower is better.
# Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI) compare
# the clusters with the true classes, where 1.0 is a perfect match.

import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score
)

X, y = load_wine(return_X_y=True)
X = StandardScaler().fit_transform(X)
print("Dataset:", X.shape[0], "rows,", X.shape[1], "features,", len(np.unique(y)), "true classes")
print()

# Part 1: try different values of k for KMeans
print(f"{'k':>2s} {'Inertia':>9s} {'Silhouette':>11s} {'Davies-Bouldin':>15s} {'Calinski-Harabasz':>18s}")
rows = []
for k in range(2, 9):
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    inertia = labels.inertia_
    sil = silhouette_score(X, labels.labels_)
    db = davies_bouldin_score(X, labels.labels_)
    ch = calinski_harabasz_score(X, labels.labels_)
    rows.append((k, inertia, sil, db, ch))
    print(f"{k:2d} {inertia:9.1f} {sil:11.4f} {db:15.4f} {ch:18.1f}")

best_sil = max(rows, key=lambda r: r[2])[0]
best_db = min(rows, key=lambda r: r[3])[0]
best_ch = max(rows, key=lambda r: r[4])[0]
print()
print("Best k by silhouette:", best_sil)
print("Best k by Davies-Bouldin:", best_db)
print("Best k by Calinski-Harabasz:", best_ch)

# Part 2: compare three algorithms using the true number of classes
k = 3
models = {
    "KMeans": KMeans(n_clusters=k, n_init=10, random_state=42),
    "Agglomerative (ward)": AgglomerativeClustering(n_clusters=k, linkage="ward"),
    "Gaussian Mixture": GaussianMixture(n_components=k, random_state=42),
}
print()
print(f"{'Algorithm':22s} {'Silhouette':>11s} {'ARI':>7s} {'NMI':>7s}")
scores = {}
for name, model in models.items():
    labels = model.fit_predict(X)
    sil = silhouette_score(X, labels)
    ari = adjusted_rand_score(y, labels)
    nmi = normalized_mutual_info_score(y, labels)
    scores[name] = ari
    print(f"{name:22s} {sil:11.4f} {ari:7.4f} {nmi:7.4f}")

# Summary
best_algo = max(scores, key=scores.get)
print()
print("Summary:")
print(f"The three internal metrics pick k = {best_sil}, {best_db} and {best_ch}.")
print("The true number of classes is 3.")
if len({best_sil, best_db, best_ch}) > 1:
    print("The metrics do not fully agree, so no single metric should be trusted alone.")
else:
    print("All three metrics agree on the same k.")
print(f"{best_algo} matches the true classes best (ARI {scores[best_algo]:.3f}).")
print("Internal metrics measure cluster shape, not correctness, so a high")
print("silhouette score does not guarantee that the clusters are meaningful.")
