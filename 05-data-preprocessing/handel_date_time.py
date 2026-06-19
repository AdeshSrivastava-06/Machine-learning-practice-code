import pandas as pd
import numpy as np

date = pd.read_csv('orders.csv')
time = pd.read_csv('messages.csv')

#print(date.head())
#print(time.head())

"""by-defaule the pandas store the date and time as object so to perform any function we should convert it int integers"""

date['date'] = pd.to_datetime(date['date'])
print(date.info())

date['date_year'] = date['date'].dt.year
date['date_month_no'] = date['date'].dt.month
date['date_month_name'] = date['date'].dt.month_name()
date['date_day'] = date['date'].dt.day
date['date_dow'] = date['date'].dt.dayofweek
date['date_dow_name'] = date['date'].dt.day_name()
date['date_is_weekend'] = np.where(date['date_dow_name'].isin(['Sunday','Saturday']),1,0)
print(date.drop(columns=['product_id','city_id','orders']).head())


import datetime

today = datetime.datetime.today()

print((today - date['date']).dt.days)

"""time"""

time['date'] = pd.to_datetime(time['date'])

time['hour'] = time['date'].dt.hour
time['min'] = time['date'].dt.minute
time['sec'] = time['date'].dt.second
time['time'] = time['date'].dt.time

print(time.head())