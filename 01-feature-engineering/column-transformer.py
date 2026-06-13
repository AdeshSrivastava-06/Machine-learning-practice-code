import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

df = pd.read_csv('covid_toy.csv')

print(df.head())
"""here the gender and city r nominnal categorical data->OHE and cough->ordinal and there 10 null values in fever so impute"""

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(df.drop(columns=['has_covid']),df['has_covid'],test_size=0.2)


from sklearn.compose import ColumnTransformer

transformer = ColumnTransformer(transformers=[
    ('tnf1',SimpleImputer(),['fever']),
    ('tnf2',OrdinalEncoder(categories=[['Mild','Strong']]),['cough']),
    ('tnf3',OneHotEncoder(sparse_output=False,drop='first'),['gender','city'])],remainder='passthrough')

transfromed = transformer.fit_transform(x_train)
transfromed1 = transformer.transform(x_test)
print(transfromed)
print(transfromed.shape)
print(transfromed1.shape)

"""in the simple inputer we the values are filed default by taking the avg and we can also fill the null values with stratergy ='most_frequent' if its a categorical data"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


lr = LogisticRegression(max_iter=100)
lr.fit(transfromed,y_train)
y_pred = lr.predict(transfromed1)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
