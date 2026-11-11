# Normality tests on the Breast Cancer dataset.
# Goal: check whether a feature follows a normal distribution, and see
# whether a log transform makes a skewed feature closer to normal.
# Three tests are used: Shapiro-Wilk, D'Agostino K2 and Anderson-Darling.
# For Shapiro-Wilk and D'Agostino, a p-value below 0.05 means the data is
# not normal. For Anderson-Darling, the statistic is compared with the
# critical value at the 5 percent level.

import warnings
import numpy as np
from scipy import stats
from sklearn.datasets import load_breast_cancer

# Newer SciPy versions print a warning for the Anderson-Darling test
warnings.filterwarnings("ignore", category=FutureWarning)

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
print("Dataset:", X.shape[0], "rows,", X.shape[1], "features")
print()

def normality_report(values, label):
    shapiro_p = stats.shapiro(values).pvalue
    dagostino_p = stats.normaltest(values).pvalue
    anderson = stats.anderson(values, dist="norm")
    # The critical value at index 2 belongs to the 5 percent significance level
    anderson_normal = anderson.statistic < anderson.critical_values[2]
    skew = stats.skew(values)
    verdict = "normal" if (shapiro_p > 0.05 and dagostino_p > 0.05 and anderson_normal) else "not normal"
    print(f"{label:28s} skew={skew:6.2f}  shapiro_p={shapiro_p:.4f}  "
          f"dagostino_p={dagostino_p:.4f}  anderson_ok={anderson_normal}  {verdict}")
    return verdict

# Step 1: test a few original features
print("Original features")
features = ["mean radius", "mean texture", "mean smoothness", "mean area", "area error"]
for col in features:
    normality_report(X[col].values, col)

# Step 2: apply a log transform to the features and test again
# A log transform reduces right skew, which is common in size related features
# All these features are strictly positive, so a plain log can be used
print()
print("After log transform")
improved = []
for col in features:
    before = stats.skew(X[col].values)
    transformed = np.log(X[col].values)
    after = stats.skew(transformed)
    normality_report(transformed, "log " + col)
    improved.append((col, before, after))

# Step 3: compare skewness before and after the transform
print()
print(f"{'Feature':20s} {'Skew before':>12s} {'Skew after':>11s}")
for col, before, after in improved:
    print(f"{col:20s} {before:12.2f} {after:11.2f}")

# Summary
print()
print("Summary:")
for col, before, after in improved:
    change = "reduced" if abs(after) < abs(before) else "did not reduce"
    print(f"The log transform {change} the skew of {col} ({before:.2f} to {after:.2f}).")
print("Strongly skewed features such as area error improve the most.")
print("Features that are already close to symmetric gain very little.")
print("With a large sample, even small deviations give a tiny p-value, so also")
print("look at the skew value and a QQ plot, not only the p-value.")
