# Anomaly detection with Elliptic Envelope (robust covariance).
# Goal: detect unusual samples by their Mahalanobis distance from the centre
# of the normal data, using a robust covariance estimate (Minimum Covariance
# Determinant), so that outliers do not distort the estimate.
# Data: Breast Cancer dataset. Benign tumours are treated as normal data.
# A small number of malignant tumours are mixed in as the anomalies.
# The 30 features are reduced to 10 with PCA, because the covariance matrix
# is more stable with fewer features.

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope, EmpiricalCovariance
from sklearn.metrics import roc_auc_score, precision_score, recall_score

X, y = load_breast_cancer(return_X_y=True)
rng = np.random.RandomState(42)

# In this dataset label 1 is benign and label 0 is malignant
benign = X[y == 1]
malignant = X[y == 0]
n_anomaly = 20
malignant = malignant[rng.choice(len(malignant), n_anomaly, replace=False)]

X_all = np.vstack([benign, malignant])
is_anomaly = np.array([0] * len(benign) + [1] * n_anomaly)
print("Normal samples:", len(benign), " Anomalies:", n_anomaly)
contamination = n_anomaly / len(X_all)
print(f"Contamination rate: {contamination:.3f}")
print()

X_all = StandardScaler().fit_transform(X_all)
X_all = PCA(n_components=10, random_state=42).fit_transform(X_all)

# Model 1: Elliptic Envelope with robust covariance
robust = EllipticEnvelope(contamination=contamination, random_state=42).fit(X_all)
robust_pred = (robust.predict(X_all) == -1).astype(int)
# A lower decision value means more abnormal, so the sign is flipped
robust_score = -robust.decision_function(X_all)

# Model 2: the same idea with the normal (non robust) covariance
empirical = EmpiricalCovariance().fit(X_all)
empirical_score = empirical.mahalanobis(X_all)
threshold = np.quantile(empirical_score, 1 - contamination)
empirical_pred = (empirical_score > threshold).astype(int)

print(f"{'Method':32s} {'ROC AUC':>8s} {'Precision':>10s} {'Recall':>8s}")
results = {}
for name, pred, score in [
    ("Elliptic Envelope (robust)", robust_pred, robust_score),
    ("Mahalanobis (normal covariance)", empirical_pred, empirical_score),
]:
    auc = roc_auc_score(is_anomaly, score)
    prec = precision_score(is_anomaly, pred)
    rec = recall_score(is_anomaly, pred)
    results[name] = auc
    print(f"{name:32s} {auc:8.4f} {prec:10.4f} {rec:8.4f}")

# Summary
better = max(results, key=results.get)
gap = abs(results["Elliptic Envelope (robust)"] - results["Mahalanobis (normal covariance)"])
print()
print("Summary:")
print(f"{better} has the higher ROC AUC ({results[better]:.3f}).")
if gap < 0.05:
    print(f"The gap is only {gap:.3f}. With just {n_anomaly} anomalies, this is a small difference.")
print("Elliptic Envelope fits an ellipse around the central part of the data,")
print("so a few extreme points have less effect on the fitted shape.")
print("It works best when the normal data is roughly one cluster with a")
print("bell shaped spread. It is less suitable for data with several clusters.")
