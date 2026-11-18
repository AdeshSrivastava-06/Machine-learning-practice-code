# Partial dependence and ICE (Individual Conditional Expectation) plots.
# Goal: understand how a trained model's prediction changes when a single
# feature changes, while the other features are kept as they are.
# Data: Diabetes dataset, model: Gradient Boosting Regressor.
# Note: the features in this dataset are already centred and scaled, so the
# feature values are not in their original units.
# Partial dependence shows the average effect over all samples.
# ICE shows the effect for each individual sample.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.inspection import partial_dependence, PartialDependenceDisplay

data = load_diabetes(as_frame=True)
X, y = data.data, data.target
print("Dataset:", X.shape[0], "rows, features:", list(X.columns))
print()

# Step 1: train the model and check how well it performs
model = GradientBoostingRegressor(
    n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
gb_r2 = cross_val_score(model, X, y, cv=cv, scoring='r2').mean()
lin_r2 = cross_val_score(LinearRegression(), X, y, cv=cv, scoring='r2').mean()
print(f"Gradient Boosting CV R2: {gb_r2:.4f}")
print(f"Linear Regression CV R2: {lin_r2:.4f}")
model.fit(X, y)

# Step 2: partial dependence of every feature
# The range of the partial dependence curve shows how much a feature can
# change the prediction, so it is used here to rank the features
ranges = {}
curves = {}
for col in X.columns:
    result = partial_dependence(model, X, [col], kind="both", grid_resolution=20)
    average = result["average"][0]
    ranges[col] = average.max() - average.min()
    curves[col] = (result["grid_values"][0], average, result["individual"][0])

ranking = pd.Series(ranges).sort_values(ascending=False)
print()
print("Features ranked by partial dependence range")
print(ranking.round(1))

# Step 3: look closely at the top three features
print()
top3 = list(ranking.index[:3])
lin_coef = pd.Series(LinearRegression().fit(X, y).coef_, index=X.columns)
for col in top3:
    grid, average, individual = curves[col]
    # Total change of each person's line from the start to the end of the grid.
    # If everyone changes by a similar amount, the lines are parallel and the
    # feature has a similar effect on everyone.
    change = individual[:, -1] - individual[:, 0]
    ice_spread = change.std()
    low, high = average[0], average[-1]
    direction = "increases" if high > low else "decreases"
    print(f"{col}: prediction {direction} from {low:.1f} to {high:.1f} across its range, "
          f"average change {change.mean():.1f}, spread of change across people {ice_spread:.1f}, "
          f"linear coefficient {lin_coef[col]:.0f}")

# Step 4: draw the plots for the top three features
# kind="both" draws the ICE lines for a sample of 50 people and the average line
PartialDependenceDisplay.from_estimator(
    model, X, top3, kind="both", subsample=50, grid_resolution=20, random_state=42)
plt.suptitle("Partial dependence and ICE for the top three features")
plt.tight_layout()
plt.show()

# Summary
print()
print("Summary:")
print(f"The most influential features are {', '.join(top3)}.")
print("Partial dependence shows the direction and shape of each effect, which a")
print("single importance number cannot show.")
print("ICE lines that are parallel mean the feature has a similar effect on everyone.")
print("ICE lines that cross or have different slopes mean the effect depends on")
print("other features, and the average line hides this.")
print("Lines that are only far apart vertically just mean different people have")
print("different starting predictions, which is not an interaction.")
if gb_r2 < lin_r2:
    print("Note: the boosting model scored lower than linear regression on this data.")
    print("Partial dependence only explains the model, so its curves are only as")
    print("trustworthy as the model itself.")
