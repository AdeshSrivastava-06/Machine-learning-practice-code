# Anomaly detection using the reconstruction error of an autoencoder.
# Goal: detect unusual samples with a neural network that is trained only on
# normal data. An autoencoder compresses each sample into a small code and then
# rebuilds it. After training on normal data it rebuilds normal samples well,
# but it rebuilds unusual samples badly, so a large error means an anomaly.
# Data: Digits dataset (8 x 8 images). Digits 0 to 4 are the normal data, and
# digits 5 to 9 are the anomalies, which the model never sees during training.
# PCA reconstruction error is used as a baseline, because PCA is a linear
# version of the same idea.
# The experiment is repeated with 5 different random splits and averaged.

import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)

X, y = load_digits(return_X_y=True)
X = X / 16.0
normal_X = X[y <= 4]
anomaly_X = X[y >= 5]
print("Normal samples (digits 0 to 4):", len(normal_X))
print("Anomaly samples (digits 5 to 9):", len(anomaly_X))
print()

def reconstruction_error(original, rebuilt):
    return ((original - rebuilt) ** 2).mean(axis=1)

BOTTLENECK = 8
N_ANOMALIES = 30
auc_ae, auc_pca = [], []
recall_ae, recall_pca = [], []
fpr_ae, fpr_pca = [], []
last_errors = None

for seed in range(5):
    rng = np.random.RandomState(seed)
    train, test_normal = train_test_split(normal_X, test_size=0.3, random_state=seed)
    test_anomaly = anomaly_X[rng.choice(len(anomaly_X), N_ANOMALIES, replace=False)]
    test = np.vstack([test_normal, test_anomaly])
    is_anomaly = np.array([0] * len(test_normal) + [1] * N_ANOMALIES)

    # Autoencoder: input to a small code of 8 values and back to the input
    autoencoder = MLPRegressor(hidden_layer_sizes=(32, BOTTLENECK, 32),
                               max_iter=2000, random_state=seed)
    autoencoder.fit(train, train)

    # PCA baseline with the same number of components as the bottleneck
    pca = PCA(n_components=BOTTLENECK).fit(train)

    for name, rebuild, aucs, recalls, fprs in [
        ("autoencoder", autoencoder.predict, auc_ae, recall_ae, fpr_ae),
        ("pca", lambda d: pca.inverse_transform(pca.transform(d)), auc_pca, recall_pca, fpr_pca),
    ]:
        train_error = reconstruction_error(train, rebuild(train))
        test_error = reconstruction_error(test, rebuild(test))
        # Threshold: the 95th percentile of the training error
        threshold = np.percentile(train_error, 95)
        flagged = test_error > threshold
        aucs.append(roc_auc_score(is_anomaly, test_error))
        recalls.append(flagged[is_anomaly == 1].mean())
        fprs.append(flagged[is_anomaly == 0].mean())
        if seed == 0:
            print(f"{name}: average error on normal {test_error[is_anomaly == 0].mean():.4f}, "
                  f"on anomalies {test_error[is_anomaly == 1].mean():.4f}")

print()
print(f"{'Method':14s} {'ROC AUC':>8s} {'Anomalies found':>16s} {'False alarms':>13s}")
print(f"{'Autoencoder':14s} {np.mean(auc_ae):8.4f} {np.mean(recall_ae):16.3f} {np.mean(fpr_ae):13.3f}")
print(f"{'PCA':14s} {np.mean(auc_pca):8.4f} {np.mean(recall_pca):16.3f} {np.mean(fpr_pca):13.3f}")

# Summary
better = "Autoencoder" if np.mean(auc_ae) > np.mean(auc_pca) else "PCA"
print()
print("Summary:")
print(f"{better} had the higher ROC AUC on average.")
print("Both methods give a larger reconstruction error for digits they were not trained on.")
print("The threshold is the 95th percentile of the training error, so about 5 percent")
print("of normal samples are flagged by design.")
print("An autoencoder can learn non-linear patterns, but it needs more data and")
print("tuning than PCA, and on a small dataset it may not beat the linear baseline.")
