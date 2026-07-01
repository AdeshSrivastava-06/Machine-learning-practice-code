import numpy as np
import pandas as pd

df = pd.read_csv('Life Expectancy Data.csv')
print(df.isnull().sum())
print(df.head())

"""we will replace the null values with the mean of the respective columns and for categorical columns we will replace the null values with the mode of the respective columns"""

for column in df.columns:
    if df[column].dtype == 'object':
        df[column].fillna(df[column].mode()[0],inplace=True)
    else:
        df[column].fillna(df[column].mean(),inplace=True)

from catboost import CatBoostRegressor,Pool
#pool is used to create a pool of data for training and testing

x = df.drop('Life expectancy ',axis=1)
y = df['Life expectancy ']

from sklearn.model_selection import cross_val_score, train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

pool_train = Pool(x_train,y_train,cat_features=['Country','Status','Year'])
pool_test = Pool(x_test,y_test,cat_features=['Country','Status','Year'])

import time 

start_time = time.time()

cbr = CatBoostRegressor(iterations = 100)

cbr.fit(pool_train)
y_pred = cbr.predict(x_test)

print("Time taken to train the model:",time.time()-start_time)

from sklearn.metrics import r2_score

cbr_r2 = r2_score(y_test,y_pred)
print("CatBoost Regressor R2 Score:",cbr_r2)

"""R2 score - 97.5% which is a very good score for a regression model. We can also use cross validation to get a better estimate of the model's performance"""
