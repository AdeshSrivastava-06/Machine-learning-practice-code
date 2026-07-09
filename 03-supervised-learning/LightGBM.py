import numpy as np
import pandas as pd

df = pd.read_csv('Life Expectancy Data.csv')
print(df.isnull().sum())
print(df.head())
print(df.shape)
"""we will replace the null values with the mean of the respective columns and for categorical columns we will replace the null values with the mode of the respective columns"""

for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = df[column].fillna(df[column].mode()[0])
    else:
        df[column] = df[column].fillna(df[column].mean())
        
x = df.drop('Life expectancy ',axis=1)
y = df['Life expectancy ']
object_cols = x.select_dtypes(include=['object']).columns

for col in object_cols:
    x[col] = x[col].astype('category').cat.codes

from sklearn.model_selection import cross_val_score, train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
import time

start_time = time.time()

from lightgbm import LGBMRegressor
lgbmr = LGBMRegressor()
lgbmr.fit(x_train,y_train)
y_pred = lgbmr.predict(x_test)
print("Time taken to train the model:",time.time()-start_time)
from sklearn.metrics import r2_score
lgbmr_r2 = r2_score(y_test,y_pred)
print("LightGBM Regressor R2 Score:",lgbmr_r2)

"""r2  0.9673210997806895"""
