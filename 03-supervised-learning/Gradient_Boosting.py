"""
Gradient Boosting Regressor - from scratch vs sklearn

Idea (squared-error loss):
    1. Start with a constant prediction F0 = mean(y)
    2. Compute residuals r = y - F(x)  (the negative gradient of the loss)
    3. Fit a small tree to the residuals
    4. Update F(x) = F(x) + learning_rate * tree(x)
    5. Repeat for n_estimators rounds
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


class MyGradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.init_pred = None

    def fit(self, X, y):
        self.trees = []
        self.init_pred = np.mean(y)
        pred = np.full(len(y), self.init_pred, dtype=float)

        for _ in range(self.n_estimators):
            residuals = y - pred
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            pred += self.learning_rate * tree.predict(X)
            self.trees.append(tree)
        return self

    def predict(self, X):
        pred = np.full(X.shape[0], self.init_pred, dtype=float)
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)
        return pred

    def staged_predict(self, X):
        """Yield predictions after each boosting round."""
        pred = np.full(X.shape[0], self.init_pred, dtype=float)
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)
            yield pred.copy()


if __name__ == "__main__":
    # Data
    X, y = make_regression(n_samples=1000, n_features=10, n_informative=6,
                           noise=15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Custom model
    my_gb = MyGradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3)
    my_gb.fit(X_train, y_train)
    my_pred = my_gb.predict(X_test)

    # sklearn model
    sk_gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1,
                                      max_depth=3, random_state=42)
    sk_gb.fit(X_train, y_train)
    sk_pred = sk_gb.predict(X_test)

    # Compare
    print(f"{'Model':<20}{'RMSE':>10}{'R2':>10}")
    for name, pred in [("Custom GB", my_pred), ("sklearn GB", sk_pred)]:
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        print(f"{name:<20}{rmse:>10.3f}{r2_score(y_test, pred):>10.3f}")

    # Error vs number of trees
    train_err = [mean_squared_error(y_train, p) for p in my_gb.staged_predict(X_train)]
    test_err = [mean_squared_error(y_test, p) for p in my_gb.staged_predict(X_test)]

    plt.figure(figsize=(8, 5))
    plt.plot(train_err, label="Train MSE")
    plt.plot(test_err, label="Test MSE")
    plt.xlabel("Number of trees")
    plt.ylabel("MSE")
    plt.title("Gradient Boosting: error vs number of trees")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    # Effect of learning rate
    plt.figure(figsize=(8, 5))
    for lr in [0.01, 0.1, 0.5]:
        m = MyGradientBoostingRegressor(n_estimators=200, learning_rate=lr, max_depth=3)
        m.fit(X_train, y_train)
        errs = [mean_squared_error(y_test, p) for p in m.staged_predict(X_test)]
        plt.plot(errs, label=f"lr={lr}")
    plt.xlabel("Number of trees")
    plt.ylabel("Test MSE")
    plt.title("Effect of learning rate")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
