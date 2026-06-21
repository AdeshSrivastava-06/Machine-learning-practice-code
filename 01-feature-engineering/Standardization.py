#standardization is a technique to change the values of numeric columns in the dataset to a common scale, without distorting differences in the ranges of values. Standardization is also called z-score normalization. It is used when we want to scale the data between -1 and 1.
#applied to features that follow a Gaussian distribution. It is also used when we want to preserve the shape of the original distribution. It is not sensitive to outliers.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

df = pd.read_csv('Social_Network_Ads.csv')
df = df.iloc[:,2:]

print(df.sample(5))

"""train-test split"""
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(df.drop('Purchased',axis=1),df['Purchased'],test_size=0.3,random_state=0)

print(x_train.shape)
print(x_test.shape)

"""standard  scalar"""

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

"""fit the scaler to the train set,it will learn the paramerters"""
scaler.fit(x_train)  #fit means-> calculating the mean and the std

"""transform train and test sets"""
x_train_scaled = scaler.fit_transform(x_train) #applying the formula z-score
x_test_scaled = scaler.transform(x_test)

print("Scaler-mean:--",scaler.mean_)

"""scaler return an numpy array"""

x_train_scaled = pd.DataFrame(x_train_scaled,columns=x_train.columns)
x_test_scaled = pd.DataFrame(x_test_scaled,columns=x_test.columns)


"""observing the difference"""
print("X-train describe:--",np.round(x_train.describe(),1))
print("X-train_scaled_describe:--",np.round(x_train_scaled.describe(),1))

"""effects of scaling"""

fig,(ax1,ax2) = plt.subplots(ncols = 2,figsize=(12,5))

ax1.scatter(x_train['Age'],x_train['EstimatedSalary'])
ax1.set_title("Before scaling")

ax2.scatter(x_train_scaled['Age'],x_train_scaled['EstimatedSalary'],color="red")
ax2.set_title("After scaling")
plt.show()

"""more effects"""

fig,(ax1,ax2) = plt.subplots(ncols=2,figsize=(12,5))
ax1.set_title("Before scaling")
sns.kdeplot(x_train['Age'],ax=ax1)
sns.kdeplot(x_train['EstimatedSalary'],ax=ax1)
ax2.set_title("After scaling")
sns.kdeplot(x_train_scaled['Age'],ax=ax2)
sns.kdeplot(x_train_scaled['EstimatedSalary'],ax=ax2)
plt.show()


"""why scaling is imp->from logistic regression"""

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

"""in decision tress there is no effect of scaling"""
"""the outiers still remain the same and their effect as well use standardization->k-means,KNN,PCA,ANN,Gradient Descent"""
