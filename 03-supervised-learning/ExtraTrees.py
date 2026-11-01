# Extra Trees (Extremely Randomized Trees) vs Random Forest
# Dataset: heart.csv (last column = target)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load data
df = pd.read_csv("heart.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Models
# RF: bootstrap=True, best split choose karta hai
# ET: bootstrap=False, random split thresholds choose karta hai (faster, lower variance)
et = ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

et.fit(X_train, y_train)
rf.fit(X_train, y_train)

# 3. Test accuracy
print("Extra Trees test accuracy :", round(accuracy_score(y_test, et.predict(X_test)), 4))
print("Random Forest test accuracy:", round(accuracy_score(y_test, rf.predict(X_test)), 4))

# 4. 5-fold cross-validation
et_cv = cross_val_score(et, X, y, cv=5)
rf_cv = cross_val_score(rf, X, y, cv=5)
print(f"\nExtra Trees CV  : {et_cv.mean():.4f} (+/- {et_cv.std():.4f})")
print(f"Random Forest CV: {rf_cv.mean():.4f} (+/- {rf_cv.std():.4f})")

print("\nClassification report (Extra Trees):")
print(classification_report(y_test, et.predict(X_test)))

# 5. Effect of number of trees
n_trees = [10, 25, 50, 100, 200, 300]
et_scores, rf_scores = [], []
for n in n_trees:
    et_scores.append(cross_val_score(
        ExtraTreesClassifier(n_estimators=n, random_state=42, n_jobs=-1), X, y, cv=5).mean())
    rf_scores.append(cross_val_score(
        RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1), X, y, cv=5).mean())

plt.figure(figsize=(7, 4))
plt.plot(n_trees, et_scores, marker="o", label="Extra Trees")
plt.plot(n_trees, rf_scores, marker="s", label="Random Forest")
plt.xlabel("Number of trees")
plt.ylabel("CV accuracy")
plt.title("Extra Trees vs Random Forest")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 6. Feature importance
imp = pd.Series(et.feature_importances_, index=X.columns).sort_values()
imp.plot(kind="barh", figsize=(7, 5), title="Extra Trees - Feature Importance")
plt.tight_layout()
plt.show()
