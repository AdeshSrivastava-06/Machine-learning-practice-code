import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

KEEP_FRAC = 0.15  # fraction of class 1 rows to keep

df = pd.read_csv("heart.csv")
print("Original distribution:", Counter(df["target"]))

pos = df[df["target"] == 1].sample(frac=KEEP_FRAC, random_state=42)
neg = df[df["target"] == 0]
df = pd.concat([pos, neg]).sample(frac=1, random_state=42)
print("Imbalanced distribution:", Counter(df["target"]))

X = df.drop(columns="target")
y = df["target"]

samplers = {
    "Baseline": None,
    "Class weights": None,
    "Random oversample": RandomOverSampler(random_state=42),
    "SMOTE": SMOTE(random_state=42),
    "Undersample": RandomUnderSampler(random_state=42),
}

models = {
    "Logistic Regression": lambda cw: LogisticRegression(
        max_iter=1000, class_weight=cw),
    "Random Forest": lambda cw: RandomForestClassifier(
        n_estimators=200, class_weight=cw, random_state=42, n_jobs=-1),
}

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
scoring = {
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": "recall",
    "f1": "f1",
    "pr_auc": "average_precision",
}

results = {}
for model_name, model_fn in models.items():
    rows = {}
    for name, sampler in samplers.items():
        cw = "balanced" if name == "Class weights" else None

        steps = [("scale", StandardScaler())]
        if sampler is not None:
            steps.append(("resample", sampler))
        steps.append(("model", model_fn(cw)))

        res = cross_validate(Pipeline(steps), X, y, cv=cv, scoring=scoring)
        rows[name] = {
            "Precision": np.mean(res["test_precision"]),
            "Recall": np.mean(res["test_recall"]),
            "F1": np.mean(res["test_f1"]),
            "PR-AUC": np.mean(res["test_pr_auc"]),
        }

    results[model_name] = pd.DataFrame(rows).T.round(3)
    print(f"\n{model_name}")
    print(results[model_name])

pd.DataFrame({m: r["F1"] for m, r in results.items()}).plot(
    kind="bar", figsize=(9, 5))
plt.ylabel("F1")
plt.title("F1 by strategy (50 CV splits)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()
