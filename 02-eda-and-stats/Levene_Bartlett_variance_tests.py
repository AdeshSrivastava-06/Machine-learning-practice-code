# Equal variance tests on the Iris dataset.
# Goal: check whether the spread of a feature is similar across the three
# species, and choose the right group comparison test based on the result.
# Levene test works well even when data is not normal.
# Bartlett test is more powerful but needs the data to be normal.
# A p-value below 0.05 means the variances are not equal.

import numpy as np
from scipy import stats
from sklearn.datasets import load_iris

data = load_iris(as_frame=True)
df = data.frame
df["species"] = df["target"].map(dict(enumerate(data.target_names)))
features = data.feature_names
print("Dataset:", df.shape[0], "rows, 3 species with 50 samples each")
print()

# Step 1: test equal variance for every feature across the species
print(f"{'Feature':22s} {'Levene p':>10s} {'Bartlett p':>11s}  Equal variance")
results = {}
for col in features:
    groups = [g[col].values for _, g in df.groupby("species")]
    levene_p = stats.levene(*groups).pvalue
    bartlett_p = stats.bartlett(*groups).pvalue
    equal = levene_p > 0.05
    results[col] = equal
    print(f"{col:22s} {levene_p:10.4f} {bartlett_p:11.4f}  {equal}")

# Step 2: choose the comparison test based on the variance result
# Standard ANOVA assumes equal variance. Welch ANOVA does not.
# Welch ANOVA is computed here with the formula, since scipy has no direct function.
def welch_anova(groups):
    k = len(groups)
    n = np.array([len(g) for g in groups])
    means = np.array([g.mean() for g in groups])
    variances = np.array([g.var(ddof=1) for g in groups])
    w = n / variances
    weighted_mean = (w * means).sum() / w.sum()
    a = (w * (means - weighted_mean) ** 2).sum() / (k - 1)
    b = 1 + (2 * (k - 2) / (k ** 2 - 1)) * ((1 - w / w.sum()) ** 2 / (n - 1)).sum()
    f_stat = a / b
    df2 = (k ** 2 - 1) / (3 * ((1 - w / w.sum()) ** 2 / (n - 1)).sum())
    p_value = stats.f.sf(f_stat, k - 1, df2)
    return f_stat, p_value

print()
print("Group comparison test chosen for each feature")
for col in features:
    groups = [g[col].values for _, g in df.groupby("species")]
    if results[col]:
        stat, p = stats.f_oneway(*groups)
        name = "Standard ANOVA"
    else:
        stat, p = welch_anova(groups)
        name = "Welch ANOVA"
    print(f"{col:22s} {name:15s} statistic={stat:8.2f}  p={p:.2e}")

# Summary
unequal = [c for c, e in results.items() if not e]
print()
print("Summary:")
print("Features with unequal variance across species:", unequal)
print("For these features, standard ANOVA is not reliable, so Welch ANOVA is used.")
print("Checking variance first avoids using a test whose assumption is broken.")
