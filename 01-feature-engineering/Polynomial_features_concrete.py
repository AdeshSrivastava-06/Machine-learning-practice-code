# Polynomial and interaction features on the concrete strength dataset.
# Goal: check whether squared and interaction terms improve a linear model.
# The last column of concrete_data.csv is assumed to be the target (strength).
# Every setup is scored with 5-fold cross validation using R2.

import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge

# Load the dataset (keep the CSV in the same folder as this script)
df = pd.read_csv("concrete_data.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
print("Dataset:", X.shape[0], "rows")
print("Features:", list(X.columns))
print("Target:", y.name)
print()

# The rows in this file are not in random order, so the folds must be shuffled.
# Without shuffling, each fold would contain a different kind of sample.
cv = KFold(n_splits=5, shuffle=True, random_state=42)

def cv_r2(model):
    return cross_val_score(model, X, y, cv=cv, scoring="r2").mean()

n = X.shape[1]
n_inter = n + n * (n - 1) // 2
n_deg2 = n + n * (n + 1) // 2
n_deg3 = PolynomialFeatures(3, include_bias=False).fit(X).n_output_features_

# Setup 1: original features only
base = cv_r2(make_pipeline(StandardScaler(), LinearRegression()))

# Setup 2: original features plus interaction terms
inter = cv_r2(make_pipeline(
    PolynomialFeatures(2, interaction_only=True, include_bias=False),
    StandardScaler(), Ridge(alpha=1)))

# Setup 3: original features plus squares and interaction terms
deg2 = cv_r2(make_pipeline(
    PolynomialFeatures(2, include_bias=False),
    StandardScaler(), Ridge(alpha=1)))

# Setup 4: degree 3 with Ridge
deg3 = cv_r2(make_pipeline(
    PolynomialFeatures(3, include_bias=False),
    StandardScaler(), Ridge(alpha=10)))

print(f"{'Setup':40s} {'Features':>8s} {'CV R2':>9s}")
print(f"{'Original features (Linear Regression)':40s} {n:8d} {base:9.4f}")
print(f"{'With interactions (Ridge)':40s} {n_inter:8d} {inter:9.4f}")
print(f"{'With squares and interactions (Ridge)':40s} {n_deg2:8d} {deg2:9.4f}")
print(f"{'Degree 3 (Ridge)':40s} {n_deg3:8d} {deg3:9.4f}")

# Find the new features with the largest coefficients in the degree 2 model
pf = PolynomialFeatures(2, include_bias=False)
X_poly = pf.fit_transform(StandardScaler().fit_transform(X))
names = pf.get_feature_names_out(X.columns)
coef = pd.Series(Ridge(alpha=1).fit(X_poly, y).coef_, index=names)
# Keep only the new features, which are all names that are not original columns
new_features = coef[~coef.index.isin(X.columns)]
top5 = new_features.reindex(new_features.abs().sort_values(ascending=False).index).head(5)
print()
print("Top 5 new features by coefficient size:")
print(top5.round(2))

# Summary
best_name, best_score = max(
    [("interactions", inter), ("squares and interactions", deg2), ("degree 3", deg3)],
    key=lambda item: item[1],
)
print()
print("Summary:")
if best_score > base:
    print(f"Adding {best_name} improved R2 from {base:.3f} to {best_score:.3f}.")
    print("The relationship between inputs and strength is not purely linear.")
else:
    print(f"Polynomial features did not help. R2 changed from {base:.3f} to {best_score:.3f}.")
print("Ridge regularization is used so that the extra features do not cause overfitting.")
