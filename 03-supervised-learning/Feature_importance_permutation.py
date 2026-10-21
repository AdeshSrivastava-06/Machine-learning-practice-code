"""
Permutation Feature Importance vs Built-in (Impurity) Importance

Built-in feature_importances_ (Random Forest):
    - Based on how much each feature reduces impurity across all trees.
    - Computed on TRAINING data only, so it can reward features the
      model merely memorized (even pure noise).
    - Biased toward features with many unique values.

Permutation importance:
    - Shuffle ONE feature's values on the TEST set and see how much
      the score drops. Big drop = model really depends on that feature.
    - Computed on unseen data, so it reflects what helps generalization.
    - Model-agnostic: works for any fitted model.

We add two pure random noise columns to a real dataset as a sanity
check: a good importance method should rank them near the bottom.
(On this dataset the built-in method handles noise fine, so the bigger
difference you will see is how the two methods treat CORRELATED
features: radius/perimeter/area carry the same information, so
permuting one is cheap, the model just leans on the others.)

Dataset: sklearn Breast Cancer (569 samples, 30 features).
"""

import numpy as np
import pandas as pd
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


# 1. Load data + add a pure noise feature

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

rng = np.random.RandomState(42)
X["random_noise"] = rng.normal(size=len(X))      # no relation to target
X["random_noise_2"] = rng.randint(0, 1000, len(X))  # many unique values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)


# 2. Train model

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
print(f"Train accuracy: {model.score(X_train, y_train):.4f}")
print(f"Test accuracy:  {model.score(X_test, y_test):.4f}\n")


# 3. Built-in importance

builtin = pd.Series(model.feature_importances_, index=X.columns)


# 4. Permutation importance (on TEST set)

perm = permutation_importance(
    model, X_test, y_test,
    n_repeats=20, random_state=42, scoring="accuracy", n_jobs=-1
)
perm_mean = pd.Series(perm.importances_mean, index=X.columns)
perm_std = pd.Series(perm.importances_std, index=X.columns)


# 5. Compare rankings

comparison = pd.DataFrame({
    "builtin_importance": builtin,
    "builtin_rank": builtin.rank(ascending=False).astype(int),
    "perm_importance": perm_mean,
    "perm_rank": perm_mean.rank(ascending=False).astype(int),
}).sort_values("perm_rank")

print("Top 10 features by permutation importance:")
print(comparison.head(10).round(4), "\n")

print("Noise features (should rank near the bottom):")
print(comparison.loc[["random_noise", "random_noise_2"]].round(4))


# 6. Visualize top 10 from each method

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

top_builtin = builtin.sort_values().tail(10)
colors_b = ["red" if "random" in f else "steelblue" for f in top_builtin.index]
axes[0].barh(top_builtin.index, top_builtin.values, color=colors_b)
axes[0].set_title("Built-in importance (train-based)")
axes[0].set_xlabel("Importance")

top_perm = perm_mean.sort_values().tail(10)
colors_p = ["red" if "random" in f else "darkorange" for f in top_perm.index]
axes[1].barh(top_perm.index, top_perm.values,
             xerr=perm_std[top_perm.index].values, color=colors_p, capsize=3)
axes[1].set_title("Permutation importance (test-based)")
axes[1].set_xlabel("Drop in accuracy when shuffled")

plt.suptitle("Top 10 features (red = random noise feature)")
plt.tight_layout()
plt.savefig("permutation_vs_builtin_importance.png", dpi=120)
plt.show()
