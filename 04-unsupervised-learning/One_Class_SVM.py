"""
One-Class SVM - Unsupervised Anomaly Detection
Idea: Learn a boundary around "normal" data. Points outside it are anomalies.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# Create data
rng = np.random.RandomState(42)
X_normal, _ = make_blobs(n_samples=300, centers=[[0, 0], [4, 4]],
                         cluster_std=0.8, random_state=42)
X_outliers = rng.uniform(low=-6, high=10, size=(20, 2))
X = np.vstack([X_normal, X_outliers])

# Scale the data (SVMs are sensitive to scale)
X = StandardScaler().fit_transform(X)

# Fit One-Class SVM
# nu = upper bound on fraction of outliers, gamma = RBF kernel width
model = OneClassSVM(kernel="rbf", gamma=0.5, nu=0.06)
model.fit(X)

# Predict: +1 = normal, -1 = anomaly
labels = model.predict(X)
print("Normal points :", np.sum(labels == 1))
print("Anomalies     :", np.sum(labels == -1))

# Plot decision boundary
xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 500),
                     np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 500))
Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.Blues_r)
plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors="red")  # boundary
plt.scatter(X[labels == 1, 0], X[labels == 1, 1], c="green", s=20, label="Normal")
plt.scatter(X[labels == -1, 0], X[labels == -1, 1], c="black", marker="x", s=50, label="Anomaly")
plt.title("One-Class SVM Anomaly Detection")
plt.legend()
plt.show()
