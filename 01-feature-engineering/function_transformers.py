#function transformers are used for applying the function to the data and then transforming it to the new data set so that we can use it for training and testing the model
#it can be used for log transform, square root transform, reciprocal transform, etc. it is used for feature engineering and data preprocessing
#it makes the data more normal distribution so that we can use it for training and testing the model and also for improving the accuracy of the model

import pandas as pd
import numpy as np

import scipy.stats as stats 

import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer

df = pd.read_csv('train.csv',usecols=['Age','Fare','Survived'])

df['Age'] = df['Age'].fillna(df['Age'].mean())

x = df.iloc[:,1:3]
y = df.iloc[:,0]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


plt.figure(figsize=(14,4))
plt.subplot(121)
sns.distplot(x_train['Age'])
plt.title("Age PDF")

plt.subplot(122)
stats.probplot(x_train['Age'],dist="norm",plot=plt)
plt.title("Age QQ plot")

plt.show()

plt.figure(figsize=(14,4))
plt.subplot(121)
sns.distplot(x_train['Fare'])
plt.title('Fare PDF')

plt.subplot(122)
stats.probplot(x_train['Fare'], dist="norm", plot=plt)
plt.title('Fare QQ Plot')

plt.show()


clf = LogisticRegression()
clf2 = DecisionTreeClassifier()

clf.fit(x_train,y_train)
clf2.fit(x_train,y_train)

y_pred = clf.predict(x_test)
y_pred1 = clf2.predict(x_test)

print("Accuarcy LR:--",accuracy_score(y_test,y_pred))
print("Accuracy DT:--",accuracy_score(y_test,y_pred1))

"""now applying the transfrom"""

trf = FunctionTransformer(func=np.log1p) # here np.log1p is used because it always add 1 value 

x_train_transfomed = trf.fit_transform(x_train)
x_test_transformed = trf.transform(x_test)

clf = LogisticRegression()
clf2 = DecisionTreeClassifier()

clf.fit(x_train_transfomed,y_train)
clf2.fit(x_train_transfomed,y_train)

y_pred = clf.predict(x_test_transformed)
y_pred1 = clf2.predict(x_test_transformed)

print("Accuarcy LR transformed:--",accuracy_score(y_test,y_pred))
print("Accuracy DT transformed:--",accuracy_score(y_test,y_pred1))
print("Accuracy LR transformed CV:--",np.mean(cross_val_score(clf,x_train_transfomed,y_train,scoring='accuracy',cv=10)))
print("Accuracy DT transformed CV:--",np.mean(cross_val_score(clf2,x_train_transfomed,y_train,scoring='accuracy',cv=10)))

"""plot to see the difference"""

plt.figure(figsize=(14,4))

plt.subplot(121)
stats.probplot(x_train['Fare'],dist='norm',plot=plt)
plt.title('Fare before log')

plt.subplot(122)
stats.probplot(x_train_transfomed['Fare'],dist='norm',plot=plt)
plt.title('Fare after log')

plt.show()


plt.figure(figsize=(14,4))

plt.subplot(121)
stats.probplot(x_train['Age'],dist='norm',plot=plt)
plt.title('Age before log')

plt.subplot(122)
stats.probplot(x_train_transfomed['Age'],dist='norm',plot=plt)
plt.title('Age after log')

plt.show()
"""sometimes it becomes worse then the first one after applying log-transform so evry col is not suitablef for that"""



"""other transformations"""

def apply_transform(transform):
    X = df.iloc[:,1:3]
    y = df.iloc[:,0]
    
    trf = ColumnTransformer([('log',FunctionTransformer(transform),['Fare'])],remainder='passthrough')
    
    X_trans = trf.fit_transform(X)
    
    clf = LogisticRegression()
    
    print("Accuracy",np.mean(cross_val_score(clf,X_trans,y,scoring='accuracy',cv=10)))
    
    plt.figure(figsize=(14,4))

    plt.subplot(121)
    stats.probplot(X['Fare'], dist="norm", plot=plt)
    plt.title('Fare Before Transform')

    plt.subplot(122)
    stats.probplot(X_trans[:,0], dist="norm", plot=plt)
    plt.title('Fare After Transform')

    plt.show()
    
    
apply_transform(np.log1p)