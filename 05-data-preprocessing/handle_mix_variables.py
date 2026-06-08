import numpy as np
import pandas as pd
import matplotlib .pyplot as plt

df = pd.read_csv('titanic.csv')
print(df.head(9))

print(df['number'].unique())

fig = df['number'].value_counts().plot.bar()
fig.set_title('Passengers travelling with')
plt.show()

"""extract numerical part"""
df['number_numerical'] = pd.to_numeric(df['number'],errors='coerce',downcast='integer')
 
"""extract categorical part"""

df['number_categorical'] = np.where(df['number_numerical'].isnull(),df['number'],np.nan)

print(df.head())

"""now for cabin and ticket"""

df['cabin_num'] = df['Cabin'].str.extract('(\d+)') #captures small numerical part

df['cabin_cat'] = df['Cabin'].str[0] #capture first letter

print(df.head())

df['ticket_num'] = df['Ticket'].apply(lambda s: s.split()[-1])
df['ticket_num'] = pd.to_numeric(df['ticket_num'],errors='coerce',downcast='integer')

df['ticket_cat'] = df['Ticket'].apply(lambda s: s.split()[0])
df['ticket_cat'] = np.where(df['ticket_cat'].str.isdigit(),np.nan,df['ticket_cat'])

df['cabin_cat'] = df['cabin_cat'].fillna('Missing')
df['ticket_cat'] = df['ticket_cat'].fillna('Missing')
df['number_categorical'] = df['number_categorical'].fillna('Missing')

df['cabin_num'] = df['cabin_num'].fillna(0.0)
df['ticket_num'] = df['ticket_num'].fillna(0.0)
df['number_numerical'] = df['number_numerical'].fillna(0.0)

print(df.head())

"""The above code is used to handle the mix variables in the titanic dataset. We have extracted the numerical and categorical parts from the 'number', 'Cabin', and 'Ticket' columns and created new columns for each part. We have also filled the missing values with 'Missing' for categorical variables and 0.0 for numerical variables."""

"""IT is the simplest way to handle mix variables. We can also use other techniques like one-hot encoding for categorical variables and scaling for numerical variables depending on the requirements of our analysis or model."""

