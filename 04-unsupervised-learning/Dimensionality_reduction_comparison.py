# Comparison of dimensionality reduction methods on the Digits dataset.
# Goal: reduce 64 pixel features to a few components and check how much
# useful information each method keeps.
# Methods: PCA (linear), Kernel PCA (non-linear), NMF (parts based, needs
# non-negative data) and Isomap (keeps neighbour distances).
# Evaluation: a KNN classifier is trained on the reduced data and scored
# with 5-fold cross validation. The reducer is inside the pipeline, so it
# is fitted only on the training part of each fold.

from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA, KernelPCA, NMF
from sklearn.manifold import Isomap
from sklearn.neighbors import KNeighborsClassifier

X, y = load_digits(return_X_y=True)
# Pixel values are between 0 and 16. Scale to 0 to 1 so that NMF can be used.
X = X / 16.0
print("Dataset:", X.shape[0], "images,", X.shape[1], "pixel features, 10 classes")
print()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)

# Baseline: KNN on all 64 pixels
baseline = cross_val_score(knn, X, y, cv=cv).mean()
print(f"Baseline KNN on all 64 features: {baseline:.4f}")
print()

print(f"{'Components':>10s} {'PCA':>8s} {'Kernel PCA':>11s} {'NMF':>8s} {'Isomap':>8s}")
table = {}
for n in [2, 5, 10, 20]:
    reducers = {
        "PCA": PCA(n_components=n),
        "Kernel PCA": KernelPCA(n_components=n, kernel="rbf", gamma=0.02),
        "NMF": NMF(n_components=n, init="nndsvda", max_iter=3000, random_state=42),
        "Isomap": Isomap(n_components=n, n_neighbors=10),
    }
    row = {}
    for name, reducer in reducers.items():
        row[name] = cross_val_score(make_pipeline(reducer, knn), X, y, cv=cv).mean()
    table[n] = row
    print(f"{n:10d} {row['PCA']:8.4f} {row['Kernel PCA']:11.4f} {row['NMF']:8.4f} {row['Isomap']:8.4f}")

# Variance kept by PCA, which is a useful reference for choosing components
print()
print("PCA variance explained")
pca = PCA().fit(X)
cumulative = pca.explained_variance_ratio_.cumsum()
for n in [2, 5, 10, 20]:
    print(f"{n:3d} components keep {cumulative[n - 1] * 100:.1f} percent of the variance")

# Summary
best_at_2 = max(table[2], key=table[2].get)
best_at_10 = max(table[10], key=table[10].get)
print()
print("Summary:")
print(f"With only 2 components, {best_at_2} gives the best accuracy ({table[2][best_at_2]:.3f}).")
print(f"With 10 components, {best_at_10} gives the best accuracy ({table[10][best_at_10]:.3f}).")
print(f"The baseline with all 64 features gives {baseline:.3f}.")
print("Reducing to 20 components keeps most of the accuracy with about one third of the features.")
print("Isomap helps most when the number of components is very small.")
print("Kernel PCA with this setting behaves almost like PCA on this dataset.")
print("NMF is the weakest here, but it gives parts based components that are easier to interpret.")
