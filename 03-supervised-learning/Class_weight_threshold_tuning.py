# Handling imbalanced classes with class weights and threshold tuning.
# Data: Breast Cancer dataset. All benign samples are kept, but only 30
# malignant samples are kept, so the rare class is about 8 percent.
# The rare class (malignant) is the positive class here.
# Three setups are compared with Logistic Regression:
#   1. Default model with the 0.5 threshold
#   2. class_weight="balanced" with the 0.5 threshold
#   3. Default model with a threshold chosen to maximise F1 on training data
# The test set is small, so the experiment is repeated with 20 different splits
# and the results are averaged.

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

X, y = load_breast_cancer(return_X_y=True)
# In the original data 0 is malignant and 1 is benign, so flip it
y = 1 - y
rng = np.random.RandomState(0)
benign_idx = np.where(y == 0)[0]
malignant_idx = rng.choice(np.where(y == 1)[0], 30, replace=False)
keep = np.concatenate([benign_idx, malignant_idx])
X, y = X[keep], y[keep]
print("Samples:", len(y), " Rare class share:", round(y.mean(), 3))
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

for seed in range(20):
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
best = max(names, key=lambda n: np.mean(results[n]["f1"]))
recall_default = np.mean(results[names[0]]["recall"])
recall_balanced = np.mean(results[names[1]]["recall"])
recall_tuned = np.mean(results[names[2]]["recall"])
print()
print("Summary:")
print(f"{best} gave the best average F1 ({np.mean(results[best]['f1']):.3f}).")
print(f"Recall of the rare class: default {recall_default:.3f}, balanced {recall_balanced:.3f}, tuned {recall_tuned:.3f}.")
if best == names[0]:
    print("The default model was already good on this data, so class weights and")
    print("threshold tuning did not improve F1. They are not always needed.")
else:
    print("Class weights or a tuned threshold improved F1 over the default model.")
print("Class weights usually raise recall and lower precision.")
print("PR AUC does not depend on the threshold, so it shows how well the model")
print("ranks the samples, and it is the fairest single number for imbalanced data.")
