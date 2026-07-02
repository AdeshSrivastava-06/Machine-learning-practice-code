#Kruskal-wallis Test is a non-parametric method for testing whether samples originate from the same distribution. It is used for comparing more than two groups and is an alternative to one-way ANOVA when the assumptions of ANOVA are not met. i.e., it does not assume normality or homogeneity of variances.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kruskal
import numpy as np

# Load Titanic dataset
df = sns.load_dataset("titanic")

# Use Pclass (1,2,3) as groups and Age as numeric variable
group1 = df[df['pclass'] == 1]['age'].dropna()
group2 = df[df['pclass'] == 2]['age'].dropna()
group3 = df[df['pclass'] == 3]['age'].dropna()

# 1. Visualization
plt.figure(figsize=(8,5))
sns.boxplot(x='pclass', y='age', data=df)
plt.title("Age distribution across Pclass")
plt.show()

# 2. Kruskal-Wallis Test
stat, p = kruskal(group1, group2, group3)
print("Kruskal-Wallis H Statistic:", stat)
print("p-value:", p)

if p < 0.05:
    print("Reject null hypothesis → At least one Pclass differs in Age")
else:
    print("Fail to reject null hypothesis → No significant difference between classes")

# 3. Effect Size (Epsilon-squared)
n_total = len(group1) + len(group2) + len(group3)
epsilon_sq = stat / (n_total - 1)
print("Epsilon-squared (Effect Size):", epsilon_sq)

# Optional: Post-hoc pairwise comparison (Dunn's test) if significant
# pip install scikit-posthocs
import scikit_posthocs as sp

posthoc = sp.posthoc_dunn([group1, group2, group3], p_adjust='bonferroni')
print("\nDunn's post-hoc test results (p-values):\n", posthoc)
