#Power Transformer is a technique used to stabilize variance and make the data more normal distribution-like. It is particularly useful when dealing with skewed data. The two most common methods for power transformation are Box-Cox and Yeo-Johnson.
#It can be applied to the features of a dataset to improve the performance of machine learning models, especially linear regression, by making the data more suitable for modeling. The PowerTransformer class from scikit-learn can be used to apply these transformations.

import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt 

import scipy.stats as stats

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

from sklearn.preprocessing import PowerTransformer

df = pd.read_csv('concrete_data.csv')

print(df.head())

print(df.isnull().sum())

print(df.describe())# to check if there is a negative value because box-cox does not work on negative values

x = df.drop(columns=['Strength'])
y = df.iloc[:,-1]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

"""applying regression without any transformation"""

lr = LinearRegression()

lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)

print(r2_score(y_test,y_pred))

"""cross-check with cross val score"""
print(np.mean(cross_val_score(lr,x,y,scoring='r2')))

# Plotting the distplots without any transformation

"""
for col in x_train.columns:
    plt.figure(figsize=(14,4))
    plt.subplot(121)
    sns.distplot(x_train[col])
    plt.title(col)

    plt.subplot(122)
    stats.probplot(x_train[col], dist="norm", plot=plt)
    plt.title(col)

    plt.show()"""
    
"""applying box-cox transform"""

pt = PowerTransformer(method='box-cox')

x_train_transformed = pt.fit_transform(x_train + 0.000000001)
x_test_transformed = pt.transform(x_test + 0.000000001)

lambda1 = pd.DataFrame({'cols': x_train.columns,'box_cox_lambdas':pt.lambdas_})

print(lambda1)

"""applying linear regression on transformed data"""

lr = LinearRegression()
lr.fit(x_train_transformed,y_train)

y_pred2 = lr.predict(x_test_transformed)

print(r2_score(y_test,y_pred2))

"""using cross-val-score"""

pt = PowerTransformer(method='box-cox')
x_transformed = pt.fit_transform(x + 0.0000001)

lr = LinearRegression()
print(np.mean(cross_val_score(lr,x_transformed,y,scoring='r2')))


# Before and after comparision for Box-Cox Plot
x_train_transformed = pd.DataFrame(x_train_transformed,columns=x_train.columns)

'''for col in x_train_transformed.columns:
    plt.figure(figsize=(14,4))
    plt.subplot(121)
    sns.distplot(x_train[col])
    plt.title(col)

    plt.subplot(122)
    sns.distplot(x_train_transformed[col])
    plt.title(col)

    plt.show()'''
    
    
    
"""applying yeo-johnson transform"""

pt1 = PowerTransformer()

x_train_transformed2 = pt1.fit_transform(x_train)
x_test_transformed2 = pt1.transform(x_test)

lr = LinearRegression()
lr.fit(x_train_transformed2,y_train)

y_pred3 = lr.predict(x_test_transformed2)

print(r2_score(y_test,y_pred3))

"""using cross-val-score"""

pt = PowerTransformer(method="yeo-johnson")
x_transformed2 = pt.fit_transform(x + 0.0000001)

lr = LinearRegression()
print(np.mean(cross_val_score(lr,x_transformed2,y,scoring='r2')))

x_train_transformed2 = pd.DataFrame(x_train_transformed2,columns=x_train.columns)
# Before and after comparision for Yeo-Johnson

'''
for col in x_train_transformed2.columns:
    plt.figure(figsize=(14,4))
    plt.subplot(121)
    sns.distplot(x_train[col])
    plt.title(col)

    plt.subplot(122)
    sns.distplot(x_train_transformed2[col])
    plt.title(col)

    plt.show()'''
    
print(pd.DataFrame({'cols':x_train.columns,'box_cox_lambda':pt.lambdas_,'Yeo_johnson_lambda':pt1.lambdas_}))
