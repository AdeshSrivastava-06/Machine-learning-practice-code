"""
ROC and Precision-Recall Curves

Accuracy alone can be misleading, especially on imbalanced data.
ROC and PR curves show how a classifier performs across ALL possible
decision thresholds, not just the default 0.5 cutoff.

ROC curve: True Positive Rate vs False Positive Rate.
    - AUC (Area Under Curve) close to 1.0 = good separation between
      classes at every threshold. AUC of 0.5 = random guessing.
    - Can look overly optimistic on imbalanced datasets (lots of
      true negatives inflate it).

Precision-Recall curve: Precision vs Recall.
    - More informative than ROC when the positive class is rare
      (e.g. fraud detection, disease diagnosis) since it doesn't
      involve true negatives at all.
    - Baseline (random classifier) = the positive class ratio, NOT 0.5.

This example compares two models on both an imbalanced dataset and
overlays multiple models on the same plot.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
)

# 1. Create an imbalanced dataset (10% positive class)

X, y = make_classification(
    n_samples=2000, n_features=15, n_informative=8,
    weights=[0.9, 0.1],   # 90% class 0, 10% class 1
    flip_y=0.02, random_state=42
)

print(f"Class distribution: {np.bincount(y)} (positive ratio = {y.mean():.3f})")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 2. Train two models to compare

models = {
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

for name, model in models.items():
    model.fit(X_train, y_train)


# 3. Plot ROC curves for both models

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

for name, model in models.items():
    y_proba = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    axes[0].plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc:.3f})")

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    axes[1].plot(recall, precision, linewidth=2, label=f"{name} (AP={ap:.3f})")

    print(f"\n{name}: ROC-AUC={auc:.4f}, Average Precision={ap:.4f}")

# ROC plot formatting
axes[0].plot([0, 1], [0, 1], "k--", linewidth=1, label="Random guess (AUC=0.5)")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve")
axes[0].legend(loc="lower right")
axes[0].grid(alpha=0.3)

# PR plot formatting
positive_ratio = y_test.mean()
axes[1].axhline(positive_ratio, color="k", linestyle="--", linewidth=1,
                 label=f"Random guess (baseline={positive_ratio:.3f})")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve")
axes[1].legend(loc="upper right")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("roc_pr_curves_comparison.png", dpi=120)
plt.show()

print("\nNote: on this imbalanced dataset (10% positive class), watch how")
print("ROC-AUC stays high for both models while Average Precision is more")
print("revealing of actual performance on the minority (positive) class.")
