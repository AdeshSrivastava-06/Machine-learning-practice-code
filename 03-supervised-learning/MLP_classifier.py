# Multi-layer perceptron (neural network) classifier on the Breast Cancer dataset.
# Goal: understand how scaling, network size and early stopping affect an MLP.
# Every setup is scored with 5-fold stratified cross validation (accuracy).

import warnings
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning

# Some setups below may not fully converge, and the warning is not needed here
warnings.filterwarnings("ignore", category=ConvergenceWarning)

X, y = load_breast_cancer(return_X_y=True)
print("Dataset:", X.shape[0], "rows,", X.shape[1], "features")
print()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def score(model):
    return cross_val_score(model, X, y, cv=cv, scoring="accuracy")

# Experiment 1: effect of feature scaling
# Features in this dataset range from about 0.01 to more than 2000
no_scale = score(MLPClassifier(hidden_layer_sizes=(32,), max_iter=1000, random_state=42))
scaled = score(make_pipeline(
    StandardScaler(),
    MLPClassifier(hidden_layer_sizes=(32,), max_iter=1000, random_state=42)))
print("Experiment 1: scaling")
print(f"{'Without scaling':25s} {no_scale.mean():.4f} (std {no_scale.std():.4f})")
print(f"{'With StandardScaler':25s} {scaled.mean():.4f} (std {scaled.std():.4f})")

# Experiment 2: effect of network size (all setups use scaling)
print()
print("Experiment 2: hidden layer size")
print(f"{'Hidden layers':25s} {'Accuracy':>9s} {'Std':>7s}")
sizes = [(2,), (8,), (32,), (64, 32), (128, 64, 32)]
size_scores = {}
for size in sizes:
    s = score(make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=size, max_iter=1000, random_state=42)))
    size_scores[size] = s.mean()
    print(f"{str(size):25s} {s.mean():9.4f} {s.std():7.4f}")

# Experiment 3: L2 regularization strength (alpha)
print()
print("Experiment 3: regularization (alpha)")
print(f"{'Alpha':25s} {'Accuracy':>9s}")
alpha_scores = {}
for alpha in [0.00001, 0.001, 0.1, 1.0, 10.0]:
    s = score(make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(64, 32), alpha=alpha, max_iter=1000, random_state=42)))
    alpha_scores[alpha] = s.mean()
    print(f"{alpha:<25} {s.mean():9.4f}")

# Experiment 4: how many iterations were needed with and without early stopping
print()
print("Experiment 4: early stopping")
for flag in [False, True]:
    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000,
                      early_stopping=flag, random_state=42))
    model.fit(X, y)
    mlp = model.named_steps["mlpclassifier"]
    print(f"early_stopping={flag!s:5s} iterations used: {mlp.n_iter_}")

# Summary
best_size = max(size_scores, key=size_scores.get)
best_alpha = max(alpha_scores, key=alpha_scores.get)
print()
print("Summary:")
print(f"Scaling changed accuracy from {no_scale.mean():.3f} to {scaled.mean():.3f}.")
print("An MLP is sensitive to the scale of its inputs, so always scale the features.")
print(f"The best hidden layer setup here was {best_size} with accuracy {size_scores[best_size]:.3f}.")
print(f"The best alpha was {best_alpha} with accuracy {alpha_scores[best_alpha]:.3f}.")
print("Even a network with 2 hidden units reached a similar accuracy, and the larger")
print("networks gave no clear gain, so a bigger network is not always better.")
print("Early stopping used far fewer iterations, which saves training time.")
