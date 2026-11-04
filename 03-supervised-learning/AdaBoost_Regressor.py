# AdaBoost Regressor on Life Expectancy Data.csv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Load data
df = pd.read_csv("Life Expectancy Data.csv")
df.columns = df.columns.str.strip()          # remove extra spaces in column names
df = df.drop(columns=["Country"], errors="ignore")
df = pd.get_dummies(df, drop_first=True)     # encode 'Status' etc.
df = df.dropna()

X = df.drop(columns=["Life expectancy"])
y = df["Life expectancy"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 1. Single weak learner (decision stump-like tree)
base = DecisionTreeRegressor(max_depth=4, random_state=42)
base.fit(X_train, y_train)
base_pred = base.predict(X_test)

# 2. AdaBoost Regressor
ada = AdaBoostRegressor(
    estimator=DecisionTreeRegressor(max_depth=4),
    n_estimators=100,
    learning_rate=0.5,
    loss="linear",        # options: 'linear', 'square', 'exponential'
    random_state=42,
)
ada.fit(X_train, y_train)
ada_pred = ada.predict(X_test)

# Compare
print("Single Tree  -> R2:", round(r2_score(y_test, base_pred), 4),
      "| RMSE:", round(mean_squared_error(y_test, base_pred) ** 0.5, 4))
print("AdaBoost     -> R2:", round(r2_score(y_test, ada_pred), 4),
      "| RMSE:", round(mean_squared_error(y_test, ada_pred) ** 0.5, 4))

# Error vs number of estimators
test_r2 = [r2_score(y_test, p) for p in ada.staged_predict(X_test)]
plt.plot(range(1, len(test_r2) + 1), test_r2)
plt.xlabel("Number of estimators")
plt.ylabel("Test R2")
plt.title("AdaBoost Regressor: R2 vs n_estimators")
plt.grid(True)
plt.show()

# Top features
imp = pd.Series(ada.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 5 features:\n", imp.head(5))
