"""

SVR is the regression version of SVM. It fits a function with an
"epsilon tube" around it:

    - Points INSIDE the tube (error <= epsilon) cost nothing.
    - Only points OUTSIDE the tube (support vectors) shape the fit.

Key parameters:
    - kernel : linear / poly / rbf (how flexible the fit can be)
    - C      : penalty for points outside the tube
               (high C = tight fit, risk of overfitting)
    - epsilon: tube width (large = simpler model, fewer support vectors)
    - gamma  : for rbf, how far one point's influence reaches

SVR is scale-sensitive, so BOTH features and target are scaled here
(epsilon is in target units, so an unscaled target makes it meaningless).

Dataset: sklearn Diabetes (442 patients, 10 features, target = disease
progression one year after baseline).
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score


# 1. Load data

data = load_diabetes()
X, y = data.data, data.target
print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Features: {data.feature_names}\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


def build(regressor):
    """Scale X inside the pipeline, scale y with TransformedTargetRegressor."""
    return TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), regressor),
        transformer=StandardScaler(),
    )



# 2. Compare kernels (+ Linear Regression as baseline)

models = {
    "Linear Regression": build(LinearRegression()),
    "SVR linear": build(SVR(kernel="linear", C=1.0, epsilon=0.1)),
    "SVR poly (deg 3)": build(SVR(kernel="poly", degree=3, C=1.0, epsilon=0.1)),
    "SVR rbf": build(SVR(kernel="rbf", C=1.0, epsilon=0.1)),
}

results = {}
print("Kernel comparison (test set):")
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    results[name] = (preds, rmse, r2)
    print(f"  {name:20s} RMSE={rmse:.2f}  R2={r2:.4f}")


# 3. Tune C, epsilon, gamma with GridSearchCV

param_grid = {
    "regressor__svr__C": [0.1, 1, 10, 100],
    "regressor__svr__epsilon": [0.01, 0.1, 0.5],
    "regressor__svr__gamma": ["scale", 0.01, 0.1],
}

grid = GridSearchCV(
    build(SVR(kernel="rbf")), param_grid,
    cv=5, scoring="r2", n_jobs=-1
)
grid.fit(X_train, y_train)

best_preds = grid.predict(X_test)
best_rmse = np.sqrt(mean_squared_error(y_test, best_preds))
best_r2 = r2_score(y_test, best_preds)

print("\nGridSearchCV (rbf SVR):")
print("  Best params:", grid.best_params_)
print(f"  Best CV R2: {grid.best_score_:.4f}")
print(f"  Test RMSE={best_rmse:.2f}  R2={best_r2:.4f}")

svr_best = grid.best_estimator_.regressor_.named_steps["svr"]
print(f"  Support vectors: {len(svr_best.support_)}/{len(X_train)}")


# 4. Plot: actual vs predicted for each model

results["SVR rbf (tuned)"] = (best_preds, best_rmse, best_r2)

fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), sharex=True, sharey=True)
lims = [y_test.min() - 10, y_test.max() + 10]

for ax, (name, (preds, rmse, r2)) in zip(axes, results.items()):
    ax.scatter(y_test, preds, s=25, alpha=0.7, color="steelblue")
    ax.plot(lims, lims, "k--", linewidth=1)   # perfect prediction line
    ax.set_title(f"{name}\nR2={r2:.3f}, RMSE={rmse:.1f}")
    ax.set_xlabel("Actual")
    ax.grid(alpha=0.3)

axes[0].set_ylabel("Predicted")
plt.suptitle("Actual vs Predicted (closer to dashed line = better)", y=1.02)
plt.tight_layout()
plt.savefig("svr_actual_vs_predicted.png", dpi=120, bbox_inches="tight")
plt.show()


# 5. Epsilon tube on ONE feature (BMI) to visualize support vectors

bmi_idx = list(data.feature_names).index("bmi")
Xb_train, Xb_test = X_train[:, [bmi_idx]], X_test[:, [bmi_idx]]

svr_1d = build(SVR(kernel="rbf", C=1.0, epsilon=0.3))
svr_1d.fit(Xb_train, y_train)
svr_core = svr_1d.regressor_.named_steps["svr"]

x_line = np.linspace(Xb_train.min(), Xb_train.max(), 300).reshape(-1, 1)
y_line = svr_1d.predict(x_line)

# epsilon is in scaled-target units, so convert to original units
eps_orig = svr_core.epsilon * y_train.std()

plt.figure(figsize=(9, 6))
plt.scatter(Xb_train, y_train, s=18, color="lightgray", label="Train data")
plt.scatter(Xb_train[svr_core.support_], y_train[svr_core.support_], s=30,
            facecolors="none", edgecolors="red", label="Support vectors")
plt.plot(x_line, y_line, color="steelblue", linewidth=2, label="SVR (rbf)")
plt.fill_between(x_line.ravel(), y_line - eps_orig, y_line + eps_orig,
                 color="steelblue", alpha=0.15, label="epsilon tube")
plt.xlabel("BMI (standardized)")
plt.ylabel("Disease progression")
plt.title("SVR epsilon tube using BMI only")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("svr_epsilon_tube_bmi.png", dpi=120)
plt.show()
