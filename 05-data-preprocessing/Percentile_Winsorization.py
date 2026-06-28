"""in percentile method(we choose the lower and the upper bound(1-95th per or 5-95th)) when we use the capping then its called winsorization"""

import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt

df = pd.read_csv('weight-height.csv')

print(df.describe())

plt.figure(figsize=(12,6))

plt.subplot(121)
sns.distplot(df['Height'])
plt.subplot(122)
sns.distplot(df['Weight'])
plt.show()

plt.figure(figsize=(12,6))

plt.subplot(121)
sns.boxplot(df['Height'])
plt.subplot(122)
sns.boxplot(df['Weight'])
plt.show()

upper_limit = df['Height'].quantile(0.99)
lower_limit = df['Height'].quantile(0.01)

print("Upper-limit:-",upper_limit,
      "\nlower-limit:-",lower_limit)

"""trimming"""
new_df = df[(df['Height'] <= upper_limit) & (df['Height'] >= lower_limit)]

print(new_df)

sns.boxplot(new_df['Height'])
plt.show()

"""capping-> winsorization"""

df['Height'] = np.where(df['Height'] >= upper_limit,
        upper_limit,
        np.where(df['Height'] <= lower_limit,
        lower_limit,
        df['Height']))

print(df.describe())

plt.figure(figsize=(12,6))

plt.subplot(121)
sns.distplot(df['Height'])
plt.subplot(122)
sns.boxplot(df['Height'])
plt.show()
