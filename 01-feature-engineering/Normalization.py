#Normalization is a technique to change the values of numeric columns in the dataset to a common scale, without distorting differences in the ranges of values. Normalization is also called min-max scaling. It is used when we want to scale the data between 0 and 1.
#Applied to features that do not follow a Gaussian distribution. It is also used when we want to preserve the shape of the original distribution. It is sensitive to outliers.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('wine_data.csv', header=None, usecols=[0,1,2])
df.columns = ['Class label','Alcohol','Malic acid']

# FORCE numeric
df['Alcohol'] = pd.to_numeric(df['Alcohol'], errors='coerce')
df['Malic acid'] = pd.to_numeric(df['Malic acid'], errors='coerce')
df['Class label'] = pd.to_numeric(df['Class label'], errors='coerce')
df = df.dropna()


sns.kdeplot(data=df, x='Alcohol')
plt.show()

sns.kdeplot(data=df, x='Malic acid')
plt.show()


"""main code"""

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(df.drop('Class label',axis=1),df['Class label'],test_size=0.3,random_state=0)

print("X-train-shape",x_train.shape)
print("x-test-shape",x_test.shape)

"""min-max scaler"""
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(x_train)

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

x_train_scaled = pd.DataFrame(x_train_scaled,columns=x_train.columns)
x_test_scaled = pd.DataFrame(x_test_scaled,columns=x_test.columns)

"""in train-test-split we will always transform on both the train and test data but will fit only the train data"""

print(np.round(x_train.describe(),1))
print(np.round(x_train_scaled.describe(),1))



fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

ax1.scatter(x_train['Alcohol'], x_train['Malic acid'],c=y_train)
ax1.set_title("Before Scaling")
ax2.scatter(x_train_scaled['Alcohol'], x_train_scaled['Malic acid'],c=y_train)
ax2.set_title("After Scaling")
plt.show()



fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

# before scaling
ax1.set_title('Before Scaling')
sns.kdeplot(x_train['Alcohol'], ax=ax1)
sns.kdeplot(x_train['Malic acid'], ax=ax1)

# after scaling
ax2.set_title('After Standard Scaling')
sns.kdeplot(x_train_scaled['Alcohol'], ax=ax2)
sns.kdeplot(x_train_scaled['Malic acid'], ax=ax2)
plt.show()

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr_scaled = LogisticRegression()

lr.fit(x_train,y_train)
lr_scaled.fit(x_train_scaled,y_train)

y_pred = lr.predict(x_test)
y_pred_scaled = lr_scaled.predict(x_test_scaled)

from sklearn.metrics import accuracy_score

print("Actual:--",accuracy_score(y_test,y_pred))
print("Scaled:--",accuracy_score(y_test,y_pred_scaled))
