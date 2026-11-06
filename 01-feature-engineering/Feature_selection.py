# Feature selection on the Breast Cancer dataset.
# Goal: keep only 10 of the 30 features and check how much accuracy is lost.
# Each method picks 10 features and is scored with 5-fold cross validation.
# Selection is placed inside the pipeline so there is no data leakage.

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif, RFE, SelectFromModel
)

# Load the dataset
X, y = load_breast_cancer(return_X_y=True, as_frame=True)
print("Dataset:", X.shape[0], "rows,", X.shape[1], "features")
print()

K = 10
model = LogisticRegression(max_iter=2000)

methods = {
    "No selection (all 30)": None,
    # Filter method: scores each feature against the target independently
    "Filter - ANOVA F-test": SelectKBest(f_classif, k=K),
    "Filter - Mutual Info": SelectKBest(mutual_info_classif, k=K),
    # Wrapper method: trains a model repeatedly and removes the weakest feature
    "Wrapper - RFE": RFE(LogisticRegression(max_iter=2000), n_features_to_select=K),
    # Embedded method: uses the feature importance from a trained model
    "Embedded - Random Forest": SelectFromModel(
        RandomForestClassifier(n_estimators=200, random_state=42),
        max_features=K, threshold=-float("inf"),
    ),
}

print(f"{'Method':28s} {'Features':>8s} {'CV Accuracy':>12s}")
scores = {}
for name, selector in methods.items():
    steps = [StandardScaler()]
    if selector is not None:
        steps.append(selector)
    steps.append(model)
    accuracy = cross_val_score(make_pipeline(*steps), X, y, cv=5).mean()
    scores[name] = accuracy
    n_features = X.shape[1] if selector is None else K
    print(f"{name:28s} {n_features:8d} {accuracy:12.4f}")

# Show which features RFE selected (fitted on the full data, for display only)
rfe = RFE(LogisticRegression(max_iter=2000), n_features_to_select=K)
rfe.fit(StandardScaler().fit_transform(X), y)
print()
print("Features selected by RFE:")
print(list(X.columns[rfe.support_]))

# Summary
baseline = scores["No selection (all 30)"]
best_name = max((n for n in scores if n != "No selection (all 30)"), key=scores.get)
best_score = scores[best_name]
print()
print("Summary:")
print(f"{best_name} reached {best_score:.4f} accuracy using only {K} features.")
print(f"Using all 30 features gave {baseline:.4f}.")
print(f"Removing 20 features cost only {baseline - best_score:.4f} in accuracy.")
