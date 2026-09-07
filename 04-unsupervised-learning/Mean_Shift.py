"""
Mean Shift Clustering

Density-based, centroid-seeking clustering. Unlike K-means, you don't
need to specify the number of clusters upfront - it's discovered
automatically by finding the peaks (modes) of the data's density.

How it works: for every point, repeatedly shift it towards the average
of points within a "bandwidth" radius, until convergence. Points that
converge to the same peak belong to the same cluster.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs


def generate_data():
    centers = [[2, 2], [-2, -2], [2, -3]]
    X, y_true = make_blobs(n_samples=500, centers=centers, cluster_std=0.7, random_state=42)
    return X, y_true


def run_mean_shift(X, quantile=0.2):
    """
    bandwidth: radius of the region used to compute the shift at each step.
      - Small bandwidth -> more clusters (very local peaks)
      - Large bandwidth -> fewer clusters (broad peaks merge together)
    estimate_bandwidth() picks a reasonable value automatically based on
    the data (quantile controls how "local" that estimate is).
    """
    bandwidth = estimate_bandwidth(X, quantile=quantile, n_samples=500)

    model = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    model.fit(X)

    labels = model.labels_
    centers = model.cluster_centers_
    n_clusters = len(np.unique(labels))

    return labels, centers, n_clusters, bandwidth


def compare_bandwidths(X, quantiles=(0.1, 0.2, 0.4)):
    fig, axes = plt.subplots(1, len(quantiles), figsize=(6 * len(quantiles), 5))

    for ax, q in zip(axes, quantiles):
        labels, centers, n_clusters, bandwidth = run_mean_shift(X, quantile=q)
        ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", s=20)
        ax.scatter(centers[:, 0], centers[:, 1], c="black", marker="x", s=150, linewidths=3)
        ax.set_title(f"quantile={q}, bandwidth={bandwidth:.2f}\nclusters found={n_clusters}")

    plt.tight_layout()
    plt.savefig("mean_shift_bandwidth_comparison.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    X, y_true = generate_data()

    labels, centers, n_clusters, bandwidth = run_mean_shift(X, quantile=0.2)
    print(f"Bandwidth used: {bandwidth:.3f}")
    print(f"Number of clusters found automatically: {n_clusters}")

    compare_bandwidths(X)
