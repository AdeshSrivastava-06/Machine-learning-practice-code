import numpy as np
import pandas as pd

df = pd.read_csv('Life Expectancy Data.csv')
print(df.isnull().sum())
print(df.head())

"""we will replace the null values with the mean of the respective columns and for categorical columns we will replace the null values with the mode of the respective columns"""

for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = df[column].fillna(df[column].mode()[0])
    else:
        df[column] = df[column].fillna(df[column].mean())
        
x = df.drop('Life expectancy ',axis=1)
y = df['Life expectancy ']


#using label encoding to convert categorical variables into numerical variables
from sklearn import preprocessing

lbl = preprocessing.LabelEncoder()
x['Country'] = lbl.fit_transform(x['Country'])
x['Status'] = lbl.fit_transform(x['Status'])
x['Year'] = x['Year'].astype(int)

from sklearn.model_selection import cross_val_score, train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

import time

start_time = time.time()

from xgboost import XGBRegressor

xgbr = XGBRegressor()
xgbr.fit(x_train,y_train)
y_pred = xgbr.predict(x_test)
print("Time taken to train the model:",time.time()-start_time)
from sklearn.metrics import r2_score
xgbr_r2 = r2_score(y_test,y_pred)
print("XGBoost Regressor R2 Score:",xgbr_r2)

"""R2 : 0.9676563451055391 which is a very good score for a regression model. We can also use cross validation to get a better estimate of the model's performance"""
