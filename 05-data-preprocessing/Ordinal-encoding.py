import pandas as pd
import numpy as np

df = pd.read_csv('customer.csv')

print(df.sample(5))

df = df.iloc[:,2:]
print(df.head())

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(df.iloc[:,0:2],df.iloc[:,-1],test_size=0.2,random_state=0)
print("x-train:-",x_train)

from sklearn.preprocessing import OrdinalEncoder

oe = OrdinalEncoder(categories=[['Poor','Average','Good'],['School','UG','PG']])

oe.fit(x_train,y_train)

x_train = oe.fit_transform(x_train)
x_test = oe.transform(x_test)

print("x-train oe--",x_train)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

le.fit(y_train) #here we dont decide the order

print(le.classes_)

y_train = le.transform(y_train)
y_test = le.transform(y_test)

print(y_train)