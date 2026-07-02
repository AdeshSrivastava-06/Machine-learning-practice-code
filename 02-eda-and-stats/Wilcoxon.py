import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon

# Create synthetic "before" and "after" weights manually (non-normal looking)
before = [78, 85, 92, 70, 66, 74, 88, 90, 77, 69,
          80, 83, 91, 67, 72, 76, 81, 79, 75, 68,
          82, 84, 87, 73, 65, 71, 86, 89, 79, 77,
          70, 73, 68, 66, 80, 81, 85, 88, 76, 74,
          69, 72, 82, 84, 90, 78, 71, 67, 75, 79,
          85, 87, 83, 72, 68, 77, 81, 89, 74, 70]  # 60 numbers

after =  [76, 82, 89, 68, 64, 71, 85, 87, 75, 67,
          78, 80, 88, 65, 70, 74, 78, 77, 72, 66,
          80, 82, 85, 70, 63, 69, 83, 86, 76, 74,
          68, 70, 66, 64, 78, 79, 83, 86, 74, 72,
          66, 70, 80, 82, 88, 76, 69, 65, 73, 77,
          82, 85, 81, 70, 66, 75, 78, 87, 72, 68]

# Create DataFrame
df = pd.DataFrame({
    "person_id": range(1, 61),
    "before": before,
    "after": after
})

# Plot distributions
plt.figure(figsize=(10,5))
sns.histplot(df['before'], color='blue', label='Before', kde=True, bins=10)
sns.histplot(df['after'], color='green', label='After', kde=True, bins=10)
plt.title("Before vs After Weight Distribution")
plt.xlabel("Weight (kg)")
plt.ylabel("Count")
plt.legend()
plt.show()

# Boxplot to visualize paired change
plt.figure(figsize=(8,5))
sns.boxplot(data=df[['before','after']])
plt.title("Boxplot: Before vs After Weight")
plt.show()

# Wilcoxon test
stat, p = wilcoxon(df['before'], df['after'])

print("Wilcoxon Signed-Rank Statistic:", stat)
print("p-value:", p)

if p < 0.05:
    print("Reject null hypothesis → Significant difference between Before and After")
else:
    print("Fail to reject null hypothesis → No significant difference")


# Approximate effect size
n = len(df)
effect_size = abs(stat) / (n*(n+1)/2)
print("Effect Size (r, approximate):", effect_size)


df['change'] = df['after'] - df['before']
print("Average change:", df['change'].mean())
