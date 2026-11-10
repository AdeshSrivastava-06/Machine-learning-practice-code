# Multicollinearity check using Variance Inflation Factor (VIF).
# Goal: find features that are highly correlated with other features,
# remove them one by one, and check the effect on model performance.
# VIF of a feature is 1 / (1 - R2), where R2 comes from predicting that
# feature using all the other features. A VIF above 5 is usually a concern.
# VIF is calculated with scikit-learn, so no extra library is needed.

import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold

data = load_diabetes(as_frame=True)
X, y = data.data, data.target
print("Dataset:", X.shape[0], "rows, features:", list(X.columns))
print()

def calculate_vif(frame):
    vif = {}
    for col in frame.columns:
        others = frame.drop(columns=col)
        r2 = LinearRegression().fit(others, frame[col]).score(others, frame[col])
        vif[col] = 1 / (1 - r2)
    return pd.Series(vif).sort_values(ascending=False)

# Step 1: VIF of all features
print("VIF of all features")
print(calculate_vif(X).round(2))

# Step 2: remove the feature with the highest VIF, one at a time,
# until every remaining feature has VIF below the threshold
THRESHOLD = 5
reduced = X.copy()
removed = []
while True:
    vif = calculate_vif(reduced)
    if vif.iloc[0] <= THRESHOLD:
        break
    removed.append(vif.index[0])
    reduced = reduced.drop(columns=vif.index[0])

print()
print("Removed features in order:", removed)
print("Remaining features:", list(reduced.columns))
print()
print("VIF after removal")
print(calculate_vif(reduced).round(2))

# Step 3: compare model performance before and after removal
cv = KFold(n_splits=5, shuffle=True, random_state=42)
r2_all = cross_val_score(LinearRegression(), X, y, cv=cv, scoring="r2").mean()
r2_reduced = cross_val_score(LinearRegression(), reduced, y, cv=cv, scoring="r2").mean()

# Step 4: compare coefficient stability on two different halves of the data
half = len(X) // 2
def coef_gap(frame):
    a = LinearRegression().fit(frame.iloc[:half], y.iloc[:half]).coef_
    b = LinearRegression().fit(frame.iloc[half:], y.iloc[half:]).coef_
    return pd.Series(abs(a - b), index=frame.columns).mean()

print()
print(f"{'Setup':20s} {'Features':>8s} {'CV R2':>8s} {'Avg coef change':>16s}")
print(f"{'All features':20s} {X.shape[1]:8d} {r2_all:8.4f} {coef_gap(X):16.2f}")
print(f"{'After VIF removal':20s} {reduced.shape[1]:8d} {r2_reduced:8.4f} {coef_gap(reduced):16.2f}")

# Summary
gap_all, gap_reduced = coef_gap(X), coef_gap(reduced)
print()
print("Summary:")
print(f"Removing {len(removed)} correlated features changed R2 from {r2_all:.3f} to {r2_reduced:.3f}.")
if gap_reduced < gap_all:
    print("The coefficients are more stable between the two halves of the data.")
else:
    print("The coefficients did not become more stable on this split.")
print("Prediction quality stays almost the same, but the model is easier to interpret.")
