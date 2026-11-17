# Class weights and threshold tuning on a hard imbalanced problem.
# Data: Diabetes dataset. A patient is labelled "high risk" (positive class)
# if the disease progression value is in the top 10 percent.
# This is a hard problem because the features only partly explain the target,
# so the rare class is easy to miss.
# Three setups are compared with Logistic Regression:
#   1. Default model with the 0.5 threshold
#   2. class_weight="balanced" with the 0.5 threshold
#   3. Default model with a threshold chosen to maximise F1 on training data
# The test set is small, so the experiment is repeated with 30 different splits
# and the results are averaged.

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

X, target = load_diabetes(return_X_y=True)
cutoff = np.quantile(target, 0.90)
y = (target >= cutoff).astype(int)
print("Samples:", len(y), " High risk patients:", y.sum(), " Share:", round(y.mean(), 3))
print()

def make_model(weight=None):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, class_weight=weight))

def best_threshold(model, X_train, y_train):
    # Get out-of-fold probabilities on the training data so that the
    # threshold is chosen without looking at the test data
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    prob = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    grid = np.arange(0.05, 0.96, 0.01)
    f1s = [f1_score(y_train, prob >= t) for t in grid]
    return grid[int(np.argmax(f1s))]

names = ["Default (threshold 0.5)", "Balanced class weight", "Tuned threshold"]
results = {n: {"precision": [], "recall": [], "f1": [], "ap": []} for n in names}
thresholds = []

for seed in range(30):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed)

    default = make_model().fit(X_tr, y_tr)
    balanced = make_model("balanced").fit(X_tr, y_tr)
    t = best_threshold(make_model(), X_tr, y_tr)
    thresholds.append(t)

    p_default = default.predict_proba(X_te)[:, 1]
    p_balanced = balanced.predict_proba(X_te)[:, 1]

    setups = {
        names[0]: (p_default, 0.5),
        names[1]: (p_balanced, 0.5),
        names[2]: (p_default, t),
    }
    for name, (prob, thr) in setups.items():
        pred = (prob >= thr).astype(int)
        results[name]["precision"].append(precision_score(y_te, pred, zero_division=0))
        results[name]["recall"].append(recall_score(y_te, pred))
        results[name]["f1"].append(f1_score(y_te, pred))
        results[name]["ap"].append(average_precision_score(y_te, prob))

print(f"{'Setup':26s} {'Precision':>10s} {'Recall':>8s} {'F1':>8s} {'PR AUC':>8s}")
for name in names:
    r = results[name]
    print(f"{name:26s} {np.mean(r['precision']):10.4f} {np.mean(r['recall']):8.4f} "
          f"{np.mean(r['f1']):8.4f} {np.mean(r['ap']):8.4f}")

print()
print(f"Average tuned threshold: {np.mean(thresholds):.2f}")

# Summary
mean = lambda n, k: np.mean(results[n][k])
best = max(names, key=lambda n: mean(n, "f1"))
print()
print("Summary:")
print(f"{best} gave the best average F1 ({mean(best, 'f1'):.3f}).")
print(f"Recall of the rare class: default {mean(names[0], 'recall'):.3f}, "
      f"balanced {mean(names[1], 'recall'):.3f}, tuned {mean(names[2], 'recall'):.3f}.")
print(f"Precision: default {mean(names[0], 'precision'):.3f}, "
      f"balanced {mean(names[1], 'precision'):.3f}, tuned {mean(names[2], 'precision'):.3f}.")
if best == names[0]:
    print("The default model gave the best F1 here, so the other setups did not help.")
else:
    for name in names[1:]:
        recall_change = "higher" if mean(name, "recall") > mean(names[0], "recall") else "lower"
        precision_change = "higher" if mean(name, "precision") > mean(names[0], "precision") else "lower"
        print(f"{name} has {recall_change} recall and {precision_change} precision than the default model.")
print("PR AUC does not depend on the threshold, so it shows how well the model")
print("ranks the samples, and it is the fairest single number for imbalanced data.")
