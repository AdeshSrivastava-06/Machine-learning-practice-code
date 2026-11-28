import pandas as pd
from pandas_datareader import data as pdr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import GridSearchCV

boston = fetch_california_housing()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
print(df.head())

df['population'] = boston.target
X = df.drop('population', axis=1)
y = df['population']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeRegressor(criterion='squared_error',max_depth=5)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print("R2 Score:", r2_score(y_test, y_pred))

"""Hyperparameter Tuning using GridSearchCV"""

param_grid = {
    'max_depth': [2,4,8,10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['squared_error', 'friedman_mse']
}

grid_search = GridSearchCV(DecisionTreeRegressor(),param_grid=param_grid)
grid_search.fit(x_train, y_train)
print("Best Hyperparameters:", grid_search.best_params_)
print("Best R2 Score from GridSearchCV:", grid_search.best_score_)


