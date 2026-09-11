"""
Learning Curves

Learning curves plot training vs validation score as a function of
training set size. They're one of the fastest ways to diagnose
whether a model suffers from high bias (underfitting), high variance
(overfitting), or would benefit from more data.

How to read them:
    - Both curves converge to a LOW score, close together -> high bias
      (model too simple, more data won't help much, need a more
      complex model or better features)
    - Big GAP between train (high) and validation (low) score that
      persists -> high variance (model too complex/overfitting,
      more data or regularization would help)
    - Curves converging to a HIGH score close together -> good fit
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


def plot_learning_curve(estimator, X, y, ax, title, cv):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring="accuracy", n_jobs=-1, random_state=42
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    ax.plot(train_sizes, train_mean, "o-", color="steelblue", label="Training score")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                     alpha=0.15, color="steelblue")

    ax.plot(train_sizes, val_mean, "o-", color="darkorange", label="Validation score")
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                     alpha=0.15, color="darkorange")

    gap = train_mean[-1] - val_mean[-1]
    ax.set_title(f"{title}\n(final gap = {gap:.3f})")
    ax.set_xlabel("Training set size")
    ax.set_ylabel("Accuracy")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)


if __name__ == "__main__":
    X, y = load_breast_cancer(return_X_y=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Three models chosen to show the three classic learning curve shapes:
    #   - shallow Decision Tree  -> tends to underfit (high bias)
    #   - deep Decision Tree     -> tends to overfit (high variance)
    #   - Logistic Regression    -> usually a reasonable, well-balanced fit here
    models = {
        "Decision Tree (max_depth=2) - expect high bias": DecisionTreeClassifier(max_depth=2, random_state=42),
        "Decision Tree (unrestricted) - expect high variance": DecisionTreeClassifier(random_state=42),
        "Logistic Regression - expect good fit": LogisticRegression(max_iter=5000),
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (name, model) in zip(axes, models.items()):
        print(f"Computing learning curve for: {name}")
        plot_learning_curve(model, X, y, ax, name, cv)

    plt.tight_layout()
    plt.savefig("learning_curves_comparison.png", dpi=120)
    plt.show()
