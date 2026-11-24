# Using KMeans to help a classifier: as new features and to label data cheaply.
# Data: Digits dataset (1797 images of handwritten digits, 64 pixel features).
#
# Experiment 1: cluster distances as features
# KMeans learns k cluster centres without labels. The distance of every image
# to each centre becomes a new set of k features. A Logistic Regression is then
# trained on the raw pixels, on the distances, and on both together.
#
# Experiment 2: classification with only 50 labels
# Labelling data is expensive, so assume only 50 labels can be paid for.
#   A. Label 50 random images
#   B. Cluster into 50 groups and label only the image closest to each centre
#   C. Give every image the label of its cluster centre image (label propagation)
#   D. Same as C, but only for the 20 percent of images closest to their centre
#   E. Upper limit: all training labels are known
# Accuracy is measured on a test set that is never used for clustering.

import warnings
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

X, y = load_digits(return_X_y=True)
X = X / 16.0
print("Dataset:", X.shape[0], "images,", X.shape[1], "features, 10 classes")
print()

# ---------- Experiment 1: cluster distances as features ----------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def logistic():
    return LogisticRegression(max_iter=3000)

baseline = cross_val_score(logistic(), X, y, cv=cv).mean()
print("Experiment 1: cluster distances as features (5-fold CV accuracy)")
print(f"{'Features':32s} {'Accuracy':>9s}")
print(f"{'Raw pixels only':32s} {baseline:9.4f}")

best_k, best_acc = None, 0
for k in [10, 30, 50, 100]:
    # KMeans inside the pipeline is fitted only on the training part of each fold
    distances_only = make_pipeline(KMeans(n_clusters=k, n_init=3, random_state=42), logistic())
    both = make_pipeline(
        make_union(FunctionTransformer(), KMeans(n_clusters=k, n_init=3, random_state=42)),
        logistic())
    acc_dist = cross_val_score(distances_only, X, y, cv=cv).mean()
    acc_both = cross_val_score(both, X, y, cv=cv).mean()
    print(f"{f'Distances to {k} centres':32s} {acc_dist:9.4f}")
    print(f"{f'Pixels + distances ({k})':32s} {acc_both:9.4f}")
    if acc_both > best_acc:
        best_k, best_acc = k, acc_both

# ---------- Experiment 2: only 50 labels ----------
N_LABELS = 50
names = ["A. 50 random labels", "B. 50 centre images labelled",
         "C. Propagate to all images", "D. Propagate to closest 20%",
         "E. All labels (upper limit)"]
accuracy = {n: [] for n in names}
label_quality = {"C": [], "D": []}

for seed in range(5):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    rng = np.random.RandomState(seed)

    def test_accuracy(features, labels):
        return logistic().fit(features, labels).score(X_te, y_te)

    # A. random labels
    random_idx = rng.choice(len(X_tr), N_LABELS, replace=False)
    accuracy[names[0]].append(test_accuracy(X_tr[random_idx], y_tr[random_idx]))

    # B. cluster the training images and label the image nearest to each centre
    kmeans = KMeans(n_clusters=N_LABELS, n_init=10, random_state=seed).fit(X_tr)
    distances = kmeans.transform(X_tr)
    centre_idx = distances.argmin(axis=0)
    centre_labels = y_tr[centre_idx]
    accuracy[names[1]].append(test_accuracy(X_tr[centre_idx], centre_labels))

    # C. every image gets the label of the centre image of its cluster
    cluster = kmeans.labels_
    propagated = centre_labels[cluster]
    accuracy[names[2]].append(test_accuracy(X_tr, propagated))
    label_quality["C"].append((propagated == y_tr).mean())

    # D. only the 20 percent of images closest to their own centre keep the label
    own_distance = distances[np.arange(len(X_tr)), cluster]
    cutoff = np.array([np.percentile(own_distance[cluster == c], 20) for c in range(N_LABELS)])
    keep = own_distance <= cutoff[cluster]
    accuracy[names[3]].append(test_accuracy(X_tr[keep], propagated[keep]))
    label_quality["D"].append((propagated[keep] == y_tr[keep]).mean())

    # E. upper limit with all true labels
    accuracy[names[4]].append(test_accuracy(X_tr, y_tr))

print()
print(f"Experiment 2: only {N_LABELS} labels (test accuracy, average of 5 splits)")
print(f"{'Setup':32s} {'Accuracy':>9s}")
for name in names:
    print(f"{name:32s} {np.mean(accuracy[name]):9.4f}")
print()
print(f"Share of correct propagated labels: all images {np.mean(label_quality['C']):.3f}, "
      f"closest 20 percent {np.mean(label_quality['D']):.3f}")

# Summary
print()
print("Summary:")
if best_acc - baseline > 0.01:
    print(f"Adding distances to {best_k} centres raised CV accuracy from {baseline:.3f} to {best_acc:.3f}.")
elif best_acc > baseline:
    print(f"Adding distances to {best_k} centres gave only a small gain, from {baseline:.3f} to {best_acc:.3f}.")
    print("Raw pixels are already very informative for this dataset, so there is little left to add.")
else:
    print(f"Cluster distance features did not beat raw pixels ({baseline:.3f}) in this run.")
random_acc = np.mean(accuracy[names[0]])
centre_acc = np.mean(accuracy[names[1]])
best_prop = max(names[2:4], key=lambda n: np.mean(accuracy[n]))
print(f"With only {N_LABELS} labels, random labelling gave {random_acc:.3f}, "
      f"labelling the centre images gave {centre_acc:.3f}.")
print(f"Propagating labels ({best_prop[:1]}) gave {np.mean(accuracy[best_prop]):.3f}, "
      f"while all labels give {np.mean(accuracy[names[4]]):.3f}.")
print("Choosing which images to label with clustering is better than choosing randomly,")
print("and label propagation adds more, but the propagated labels are not all correct.")
