import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import numpy as np

# Load dataset
df = sns.load_dataset("tips")

# Independent variable: sex (Male/Female)
# Numeric variable: total_bill
group1 = df[df['sex'] == 'Male']['total_bill']
group2 = df[df['sex'] == 'Female']['total_bill']

# 1. Visual check (Boxplot)
plt.figure(figsize=(8,5))
sns.boxplot(x='sex', y='total_bill', data=df)
plt.title("Total Bill distribution by Gender")
plt.show()

# 2. Mann-Whitney U Test
stat, p = mannwhitneyu(group1, group2, alternative='two-sided')
print("Mann-Whitney U Statistic:", stat)
print("p-value:", p)

if p < 0.05:
    print("Reject null hypothesis → Significant difference between groups")
else:
    print("Fail to reject null hypothesis → No significant difference")

# 3. Effect Size (Rank-Biserial Correlation)
n1 = len(group1)
n2 = len(group2)
rank_biserial = 1 - (2*stat)/(n1*n2)
print("Rank-Biserial Correlation (Effect Size):", rank_biserial)
