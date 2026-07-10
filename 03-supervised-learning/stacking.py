import pandas as pd
import numpy as np

df = pd.read_csv('Life Expectancy Data.csv')
print(df.isnull().sum())

"""we will replace the null values with the mean of the respective columns and for categorical columns we will replace the null values with the mode of the respective columns"""

for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = df[column].fillna(df[column].mode()[0])
    else:
        df[column] = df[column].fillna(df[column].mean())
        
x = df.drop('Life expectancy ',axis=1)
y = df['Life expectancy ']


from sklearn.model_selection import cross_val_score, train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

cat_cols = x_train.select_dtypes(include='object').columns

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ],
    remainder='passthrough'
)

x_train = preprocessor.fit_transform(x_train)
x_test = preprocessor.transform(x_test)
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

estimators = [
    ('dt',DecisionTreeRegressor()),
    ('rf',RandomForestRegressor()),
    ('svr',SVR())
]

stacking_regressor = StackingRegressor(estimators=estimators,final_estimator=LinearRegression())
stacking_regressor.fit(x_train,y_train)
y_pred = stacking_regressor.predict(x_test)
from sklearn.metrics import r2_score
stacking_r2 = r2_score(y_test,y_pred)
print("Stacking Regressor R2 Score:",stacking_r2)

"""Stacking Regressor R2 Score: 0.9693545122241257"""
