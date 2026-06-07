"""
FEATURE ENGINEERING: BINARIZATION

Binarization is a feature engineering technique that converts numerical values
into binary values (0 or 1) based on a threshold.

In this case, the 'family' feature is created using:
family = SibSp + Parch

Binarization converts this feature into:
0 → Passenger is traveling alone
1 → Passenger is traveling with family

Why binarization is used:
1. Captures presence or absence of a condition
2. Removes unnecessary magnitude information
3. Improves interpretability of features
4. Helps models learn simpler and more general rules

Binarization is useful when the exact numeric value is less important
than whether the value exists or not.
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

from sklearn.compose import ColumnTransformer

df = pd.read_csv('train.csv')[['Age','Fare','SibSp','Parch','Survived']]

df['family'] = df['SibSp'] + df['Parch']

df = df.dropna()
df = df.drop(columns=['SibSp','Parch'])


X = df.drop(columns=['Survived'])
y = df['Survived']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# Without binarization

clf = DecisionTreeClassifier()

clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print(accuracy_score(y_test,y_pred))

print(np.mean(cross_val_score(DecisionTreeClassifier(),X,y,cv=10,scoring='accuracy')))

"""so here i am transformin the famili col into traveling alone or not"""

"""applying binarization"""

from sklearn.preprocessing import Binarizer

trf = ColumnTransformer([
    ('bin',Binarizer(copy=False),['family'])
],remainder='passthrough')

x_train_trf = trf.fit_transform(X_train)
x_test_trf = trf.transform(X_test)

df = pd.DataFrame(x_train_trf,columns=['family','Age','Fare'])

print(df.sample(7))

clf = DecisionTreeClassifier()
clf.fit(x_train_trf,y_train)
y_pred2 = clf.predict(x_test_trf)

print(accuracy_score(y_test,y_pred2))

X_trf = trf.fit_transform(X)
print(np.mean(cross_val_score(DecisionTreeClassifier(),X_trf,y,cv=10,scoring='accuracy')))