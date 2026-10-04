"""
Chi-Square Test

Two common chi-square tests for categorical data:

1. Chi-Square Test of Independence
   Checks whether two categorical variables are related/associated,
   or independent of each other (e.g. "is smoking status related to
   lung disease?"). Works on a contingency table (cross-tab counts).

2. Chi-Square Goodness-of-Fit Test
   Checks whether observed category counts match an EXPECTED
   distribution (e.g. "is this die fair? are these category
   proportions as expected?").

In both cases: small p-value (< 0.05) -> reject the null hypothesis
(variables ARE associated / distribution does NOT match expected).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, chisquare, chi2


# 1. Chi-Square Test of Independence

# Example: is smoking status associated with developing a disease?
data = pd.DataFrame({
    "Smoker": ["Yes", "Yes", "Yes", "No", "No", "No"] * 50,
    "Disease": (["Yes"] * 3 + ["No"] * 3) * 50,
})

# add some real association via randomness so the test isn't trivial
rng = np.random.RandomState(42)
smoker = rng.choice(["Yes", "No"], size=400, p=[0.4, 0.6])
# smokers have a higher chance of disease
disease = np.where(
    smoker == "Yes",
    rng.choice(["Yes", "No"], size=400, p=[0.55, 0.45]),
    rng.choice(["Yes", "No"], size=400, p=[0.20, 0.80]),
)
df = pd.DataFrame({"Smoker": smoker, "Disease": disease})

contingency_table = pd.crosstab(df["Smoker"], df["Disease"])
print("Contingency Table (observed counts):")
print(contingency_table)

chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

print(f"\nChi-Square Statistic: {chi2_stat:.4f}")
print(f"Degrees of Freedom: {dof}")
print(f"P-value: {p_value:.6f}")
print("\nExpected counts (if Smoker and Disease were independent):")
print(pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns).round(2))

alpha = 0.05
if p_value < alpha:
    print(f"\np-value < {alpha} -> Reject null hypothesis: Smoker and Disease ARE associated.")
else:
    print(f"\np-value >= {alpha} -> Fail to reject null: no significant association found.")

# Effect size: Cramer's V (how STRONG is the association, not just significant)
n = contingency_table.sum().sum()
min_dim = min(contingency_table.shape) - 1
cramers_v = np.sqrt(chi2_stat / (n * min_dim))
print(f"Cramer's V (effect size): {cramers_v:.4f}  (0=no association, 1=perfect association)")


# 2. Chi-Square Goodness-of-Fit Test

# Example: are these dice rolls consistent with a FAIR die?
print("\n" + "=" * 60)
print("Goodness-of-Fit Test: is this die fair?")

observed_rolls = np.array([95, 102, 88, 130, 85, 100])  # counts for faces 1-6, 600 rolls
expected_rolls = np.array([100] * 6)  # fair die -> equal counts expected

chi2_gof, p_gof = chisquare(f_obs=observed_rolls, f_exp=expected_rolls)

print(f"Observed counts: {observed_rolls}")
print(f"Expected counts (fair die): {expected_rolls}")
print(f"Chi-Square Statistic: {chi2_gof:.4f}")
print(f"P-value: {p_gof:.6f}")

if p_gof < alpha:
    print(f"\np-value < {alpha} -> Reject null: the die is likely NOT fair.")
else:
    print(f"\np-value >= {alpha} -> Fail to reject null: no evidence the die is unfair.")


# 3. Visualize the chi-square distribution and where our
#    test statistics fall on it

x = np.linspace(0, 30, 500)
dof_indep = contingency_table.size - contingency_table.shape[0] - contingency_table.shape[1] + 1
y_indep = chi2.pdf(x, df=dof_indep if dof_indep > 0 else 1)
y_gof = chi2.pdf(x, df=len(observed_rolls) - 1)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(x, y_indep, color="steelblue", label=f"Chi2 distribution (df={dof})")
axes[0].axvline(chi2_stat, color="red", linestyle="--", label=f"Test statistic = {chi2_stat:.2f}")
axes[0].fill_between(x, y_indep, where=(x >= chi2_stat), color="red", alpha=0.2, label=f"p-value region")
axes[0].set_title("Test of Independence: Smoker vs Disease")
axes[0].set_xlabel("Chi-Square value")
axes[0].set_ylabel("Density")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(x, y_gof, color="darkorange", label=f"Chi2 distribution (df={len(observed_rolls)-1})")
axes[1].axvline(chi2_gof, color="red", linestyle="--", label=f"Test statistic = {chi2_gof:.2f}")
axes[1].fill_between(x, y_gof, where=(x >= chi2_gof), color="red", alpha=0.2, label="p-value region")
axes[1].set_title("Goodness-of-Fit: Is the Die Fair?")
axes[1].set_xlabel("Chi-Square value")
axes[1].set_ylabel("Density")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("chi_square_tests.png", dpi=120)
plt.show()
