"""
Handling Skewed Data: Log & Power Transforms

Why: Linear Regression, Logistic Regression, and other algorithms that
assume normally-distributed features perform poorly on skewed data.
Transforming skewed columns toward a Gaussian shape often improves
model performance.

Techniques covered:
1. Log Transform         -> good for right-skewed data (no zeros/negatives)
2. Log1p Transform        -> like log, but handles zeros (log(1+x))
3. Square Root Transform  -> milder than log, works with zero values
4. Box-Cox Transform      -> only works on strictly positive data
5. Yeo-Johnson Transform  -> works on positive, negative, and zero values
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import PowerTransformer

# 1. Load data

df = pd.read_csv("train.csv")  # e.g. Titanic Fare column is right-skewed
col = "Fare"
data = df[col].dropna()

print("Original skewness:", data.skew())

# 2. Log transform (use log1p to safely handle 0 values)

log_transformed = np.log1p(data)
print("Log1p skewness:", log_transformed.skew())

# 3. Square root transform

sqrt_transformed = np.sqrt(data)
print("Sqrt skewness:", sqrt_transformed.skew())

# 4. Box-Cox (requires strictly positive values, no zeros)

positive_data = data[data > 0]
boxcox_transformed, best_lambda = stats.boxcox(positive_data)
print("Box-Cox skewness:", pd.Series(boxcox_transformed).skew())
print("Box-Cox best lambda:", best_lambda)

# 5. Yeo-Johnson (handles zeros/negatives too, sklearn API)

pt = PowerTransformer(method="yeo-johnson")
yj_transformed = pt.fit_transform(data.values.reshape(-1, 1)).flatten()
print("Yeo-Johnson skewness:", pd.Series(yj_transformed).skew())

# 6. Visual before/after check

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(data, bins=30)
axes[0].set_title(f"Original (skew={data.skew():.2f})")

axes[1].hist(log_transformed, bins=30)
axes[1].set_title(f"Log1p (skew={log_transformed.skew():.2f})")

axes[2].hist(yj_transformed, bins=30)
axes[2].set_title(f"Yeo-Johnson (skew={pd.Series(yj_transformed).skew():.2f})")

plt.tight_layout()
plt.savefig("skew_transform_comparison.png")
plt.show()
