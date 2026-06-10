"""
FEATURE ENGINEERING: BINNING (DISCRETIZATION)

Binning, also known as Discretization or Quantization, is a feature engineering
technique used to convert continuous numerical variables into discrete intervals
called bins. Instead of using exact numeric values, the data is represented by
bin indices.

Why binning is used:
1. Reduces noise and outliers in continuous data
2. Simplifies complex numerical distributions
3. Helps models capture non-linear relationships
4. Prevents overfitting, especially in tree-based models

Types of binning strategies:
1. Uniform: Equal width bins
2. Quantile: Equal number of samples in each bin
3. K-means: Bins created using clustering

In this code, KBinsDiscretizer is used with:
- strategy='quantile' to handle skewed data
- encode='ordinal' to assign integer labels to bins

Binning is particularly effective for Decision Trees as it creates
clear split boundaries.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.compose import ColumnTransformer

df = pd.read_csv('train.csv',usecols=['Age','Fare','Survived'])

df = df.dropna()
print(df.head())
X = df.iloc[:,1:]
y = df.iloc[:,0]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

clf = DecisionTreeClassifier()

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

print(accuracy_score(y_test,y_pred))

print(np.mean(cross_val_score(DecisionTreeClassifier(),X,y,cv=10,scoring='accuracy')))

kbin_age = KBinsDiscretizer(n_bins=18,encode='ordinal',strategy='quantile')
kbin_fare = KBinsDiscretizer(n_bins=18,encode='ordinal',strategy='quantile')

trf = ColumnTransformer([
    ('first',kbin_age,[0]),
    ('second',kbin_fare,[1])
],remainder='passthrough')

x_train_trf = trf.fit_transform(X_train)
x_test_tef = trf.transform(X_test)

print(trf.named_transformers_['first'].bin_edges_)

output = pd.DataFrame({
    'age':X_train['Age'],
    'age_trf':x_train_trf[:,0],
    'fare':X_train['Fare'],
    'fare_trf':x_train_trf[:,1]
})

output['age_labels'] = pd.cut(x=X_train['Age'],bins=trf.named_transformers_['first'].bin_edges_[0].tolist())

output['fare_labels'] = pd.cut(x=X_train['Fare'],bins=trf.named_transformers_['second'].bin_edges_
[0].tolist())

print(output.sample(5))

clf1 = DecisionTreeClassifier()

clf.fit(x_train_trf,y_train)
y_pred2 = clf.predict(x_test_tef)

print(accuracy_score(y_test,y_pred2))

x_trf = trf.fit_transform(X)
print(np.mean(cross_val_score(DecisionTreeClassifier(),x_trf,y,cv=10,scoring='accuracy')))


def discretize(bins,strategy):
    kbin_age = KBinsDiscretizer(n_bins=bins,encode='ordinal',strategy=strategy)
    kbin_fare = KBinsDiscretizer(n_bins=bins,encode='ordinal',strategy=strategy)
    
    trf = ColumnTransformer([
        ('first',kbin_age,[0]),
        ('second',kbin_fare,[1])
    ])
    
    X_trf = trf.fit_transform(X)
    print(np.mean(cross_val_score(DecisionTreeClassifier(),X,y,cv=10,scoring='accuracy')))
    
    plt.figure(figsize=(14,4))
    plt.subplot(121)
    plt.hist(X['Age'])
    plt.title("Before")

    plt.subplot(122)
    plt.hist(X_trf[:,0],color='red')
    plt.title("After")

    plt.show()
    
    plt.figure(figsize=(14,4))
    plt.subplot(121)
    plt.hist(X['Fare'])
    plt.title("Before")

    plt.subplot(122)
    plt.hist(X_trf[:,1],color='red')
    plt.title("Fare")

    plt.show()