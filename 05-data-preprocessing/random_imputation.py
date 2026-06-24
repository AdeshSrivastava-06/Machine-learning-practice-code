"""
RANDOM SAMPLE IMPUTATION

-> preserves distribution and variance
-> useful for linear models like Logistic Regression
-> memory heavy because extra feature is created
-> works only for numerical data
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = sns.load_dataset('titanic')

df = df[['age','fare','survived']]
print(df.head())

x = df.drop(columns=['survived'])
y = df['survived']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=2
)

# Random Imputation for AGE
x_train['age_imputed'] = x_train['age']
x_test['age_imputed'] = x_test['age']

"""
Logic:
1. Count missing values
2. Randomly pick same number of non-missing values
3. Assign only at NaN positions
4. Sampling is done from TRAIN data only (no data leakage)
"""

# the below code is doing the following:
# 1. Count the number of missing values in the 'age' column of the training dataset (x_train).
# 2. Randomly sample the same number of non-missing values from the 'age' column of the training dataset (x_train) using the sample() method.
# 3. The sampled values are assigned to the 'age_imputed' column of the training dataset (x_train) at the positions where the original 'age' values were missing (NaN).
random_sample_train = x_train['age'].dropna().sample(
    x_train['age'].isnull().sum(),
    replace=True,
    random_state=0
).values

# Assign the randomly sampled values to the 'age_imputed' column of the training dataset (x_train) at the positions where the original 'age' values were missing (NaN).
x_train.loc[x_train['age'].isnull(), 'age_imputed'] = random_sample_train

# test
random_sample_test = x_test['age'].dropna().sample(
    x_test['age'].isnull().sum(),
    replace=True,
    random_state=0
).values

x_test.loc[x_test['age'].isnull(), 'age_imputed'] = random_sample_test

# Distribution Check

sns.kdeplot(x_train['age'], label='Original', linewidth=2)
sns.kdeplot(x_train['age_imputed'], label='Random Imputed', linewidth=2)
plt.legend()
plt.show()

# Logistic Regression (Linear Model)
lr = LogisticRegression(max_iter=1000)

lr.fit(x_train[['age_imputed','fare']], y_train)

y_pred = lr.predict(x_test[['age_imputed','fare']])

print("Accuracy:", accuracy_score(y_test, y_pred))
