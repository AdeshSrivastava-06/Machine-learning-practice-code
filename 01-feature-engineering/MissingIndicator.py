import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.impute import MissingIndicator,SimpleImputer
df = pd.read_csv('train2.csv',usecols=['Age','Fare','Survived'])

print(df.head())

X = df.drop(columns=['Survived'])
y = df['Survived']
X_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)



si =SimpleImputer() # here the default value will be mean
x_train_trf = si.fit_transform(X_train)
x_test_trf = si.transform(x_test)

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

clf.fit(x_train_trf,y_train)

y_pred = clf.predict(x_test_trf)

from sklearn.metrics import accuracy_score
print(f'Accuracy for SimpleImputer: {accuracy_score(y_test,y_pred)}')




mi = MissingIndicator()

mi.fit(X_train)
x_train_missing = mi.transform(X_train)
x_test_missing = mi.transform(x_test)

print(x_train_missing)

X_train['Age_NA'] = x_train_missing
x_test['Age_NA'] = x_test_missing

print(X_train)


si = SimpleImputer()

x_train_trf2 = si.fit_transform(X_train)
x_test_trf2 = si.transform(x_test)

clf = LogisticRegression()
clf.fit(x_train_trf2,y_train)
y_pred = clf.predict(x_test_trf2)

print(f'Accuracy for MissingIndicator: {accuracy_score(y_test,y_pred)}')

"""in the simpleimputer we have the same function"""
si = SimpleImputer(add_indicator=True)
clf = LogisticRegression()

clf.fit(x_train_trf2,y_train)

y_pred2 = clf.predict(x_test_trf2)

print(f'Accuracy for SimpleImputer with indicator: {accuracy_score(y_test,y_pred2)}')

